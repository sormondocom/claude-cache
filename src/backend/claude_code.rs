use anyhow::{bail, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::Semaphore;

use crate::error::ProxyError;
use super::{BackendResult, ContentBlock, MessageContent, MessagesRequest, MessagesResponse, ModelBackend, Usage};

/// API backend that drives the local `claude` CLI subprocess instead of calling
/// api.anthropic.com directly.  Used automatically when ANTHROPIC_API_KEY is
/// absent (Pro/Max subscribers who only have an OAuth token), or explicitly via
/// `[api] backend = "claude_code"` in config.
///
/// Concurrency is bounded by `max_concurrency`; excess requests queue for up to
/// `queue_timeout_secs` before returning a capacity error.  Each subprocess is
/// spawned with `kill_on_drop(true)` so a timeout or cancellation never leaves
/// orphan processes behind.
pub struct ClaudeCodeBackend {
    timeout_secs:       u64,
    queue_timeout_secs: u64,
    max_concurrency:    usize,
    semaphore:          Arc<Semaphore>,
}

impl ClaudeCodeBackend {
    pub fn new(timeout_secs: u64, max_concurrency: usize, queue_timeout_secs: u64) -> Self {
        let permits = if max_concurrency == 0 {
            Semaphore::MAX_PERMITS
        } else {
            max_concurrency
        };
        ClaudeCodeBackend {
            timeout_secs,
            queue_timeout_secs,
            max_concurrency,
            semaphore: Arc::new(Semaphore::new(permits)),
        }
    }

    /// Available process slots (for logging / error messages).
    pub fn available_slots(&self) -> usize {
        self.semaphore.available_permits()
    }
}

/// Extract the text content of a single message.
fn msg_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Build the prompt string and optional system-prompt string from the request.
///
/// The last user message becomes the `claude` prompt argument.  Any prior turns
/// are folded into the system prompt as a conversation-history block so the CLI
/// sees full context without treating it as a multi-turn session.
fn build_args(req: &MessagesRequest) -> (String, Option<String>) {
    let prompt = req.messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| msg_text(&m.content))
        .unwrap_or_default();

    let history: Vec<String> = req.messages
        .iter()
        .take(req.messages.len().saturating_sub(1))
        .map(|m| {
            let label = if m.role == "user" { "Human" } else { "Assistant" };
            format!("{label}: {}", msg_text(&m.content))
        })
        .collect();

    let history_block = if history.is_empty() {
        None
    } else {
        Some(format!("[Conversation so far]\n{}", history.join("\n\n")))
    };

    let system = match (req.normalized_system(), history_block) {
        (Some(s), Some(h)) => Some(format!("{s}\n\n{h}")),
        (Some(s), None)    => Some(s),
        (None,    Some(h)) => Some(h),
        (None,    None)    => None,
    };

    (prompt, system)
}

#[async_trait]
impl ModelBackend for ClaudeCodeBackend {
    async fn complete(&self, req: &MessagesRequest) -> Result<BackendResult> {
        let start = Instant::now();

        // ── Acquire a process slot ────────────────────────────────────────────
        // This is the concurrency gate.  Excess requests wait here until a slot
        // opens or queue_timeout_secs elapses.  The permit is held for the full
        // subprocess lifetime and released when this function returns.
        let _permit = tokio::time::timeout(
            std::time::Duration::from_secs(self.queue_timeout_secs),
            self.semaphore.acquire(),
        )
        .await
        .map_err(|_| {
            let cap = if self.max_concurrency == 0 {
                "unlimited".to_string()
            } else {
                self.max_concurrency.to_string()
            };
            ProxyError::BackendAtCapacity(format!(
                "all {cap} process slot(s) occupied for >{queue}s — raise \
                 api.claude_code_max_concurrency or api.claude_code_queue_timeout_secs \
                 in config.toml",
                queue = self.queue_timeout_secs,
            ))
        })?
        // semaphore.acquire() only errors if the semaphore is closed; we never
        // close ours, so this is safe to unwrap.
        .unwrap();

        // ── Build and spawn the subprocess ────────────────────────────────────
        let (prompt, system) = build_args(req);

        let mut cmd = Command::new("claude");
        cmd.arg("--print")
           .arg("--output-format").arg("text")
           // Pass our system prompt (or empty string) via --system-prompt.
           // An explicit value — even empty — replaces the Claude Code
           // coding-assistant persona so it doesn't bleed into general Q&A.
           // Kill the child if this future is dropped (timeout / cancellation).
           // Without kill_on_drop a dropped future leaves an orphan process behind.
           .kill_on_drop(true)
           .stdin(std::process::Stdio::piped())
           .stdout(std::process::Stdio::piped())
           .stderr(std::process::Stdio::piped());

        // Always pass --system-prompt: use the caller-supplied prompt if present,
        // otherwise an empty string to suppress the default coding-assistant persona.
        let sys_arg = system.as_deref().unwrap_or("");
        cmd.arg("--system-prompt").arg(sys_arg);

        let mut child = cmd.spawn()
            .map_err(|e| ProxyError::BackendUnavailable(
                format!("failed to spawn claude CLI: {e} — is 'claude' in PATH?")
            ))?;

        // ── Write prompt via stdin ────────────────────────────────────────────
        // Using stdin avoids OS argument-length limits for large context windows.
        // We drop the handle after writing so the child can see EOF and start.
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await
                .map_err(|e| anyhow::anyhow!("writing to claude stdin: {e}"))?;
            // stdin dropped here — sends EOF to the child
        }

        // ── Wait for completion ───────────────────────────────────────────────
        // wait_with_output() consumes `child`.  If the timeout fires, the future
        // is dropped, child is dropped, and kill_on_drop(true) sends SIGKILL.
        // No orphan processes are possible regardless of how this branch exits.
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout_secs),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| ProxyError::BackendTimeout(format!(
            "claude CLI timed out after {}s — increase api.request_timeout_secs \
             for long responses",
            self.timeout_secs,
        )))?
        .map_err(|e| anyhow::anyhow!("waiting for claude: {e}"))?;

        // ── Validate exit status ──────────────────────────────────────────────
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            bail!(
                "claude exited {:?} — stderr: {stderr}  stdout: {stdout}",
                output.status.code(),
            );
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if text.is_empty() {
            bail!("claude returned an empty response");
        }

        // ── Build response ────────────────────────────────────────────────────
        // Token counts are estimated from character length (~4 chars/token) since
        // the CLI does not expose usage metadata.
        let input_tokens  = (prompt.len() as u32) / 4;
        let output_tokens = (text.len()   as u32) / 4;

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let response = MessagesResponse {
            id:          format!("cc-{ts:x}"),
            kind:        "message".to_string(),
            role:        "assistant".to_string(),
            content:     vec![ContentBlock {
                kind:  "text".to_string(),
                text:  Some(text),
                extra: Default::default(),
            }],
            model:       "claude-code".to_string(),
            stop_reason: Some("end_turn".to_string()),
            usage:       Usage { input_tokens, output_tokens },
        };

        tracing::info!(
            "claude_code backend: {}ms, ~{input_tokens} in / ~{output_tokens} out tokens \
             ({} slot(s) remaining)",
            start.elapsed().as_millis(),
            self.semaphore.available_permits(),
        );

        // _permit dropped here — releases the process slot
        Ok(BackendResult { response, confidence: None, latency_ms: start.elapsed().as_millis() as u64 })
    }

    fn name(&self) -> &'static str { "claude_code" }

    async fn is_available(&self) -> bool {
        Command::new("claude")
            .arg("--version")
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}
