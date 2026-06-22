pub mod anthropic;
pub mod claude_code;
pub mod ollama;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub use anthropic::AnthropicBackend;
pub use claude_code::ClaudeCodeBackend;
pub use ollama::OllamaBackend;

// ── Anthropic API wire types (shared by both backends) ─────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role:    String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Preserves all other block fields (e.g. `thinking`, `data`, `signature`)
    /// so thinking blocks round-trip intact through the proxy.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// The full /v1/messages request body — we preserve unknown fields for passthrough.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model:      String,
    pub messages:   Vec<Message>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system:     Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream:     Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools:      Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra:      serde_json::Map<String, serde_json::Value>,
    /// Carried from the incoming `anthropic-beta` HTTP header — not part of the JSON body.
    /// Forwarded to Anthropic so beta features (e.g. context_management) are accepted.
    #[serde(skip)]
    pub anthropic_beta: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id:           String,
    #[serde(rename = "type")]
    pub kind:         String,
    pub role:         String,
    pub content:      Vec<ContentBlock>,
    pub model:        String,
    pub stop_reason:  Option<String>,
    pub usage:        Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub input_tokens:  u32,
    pub output_tokens: u32,
}

impl MessagesRequest {
    /// Extract the concatenated text of all user messages — used for classification,
    /// exact cache keys, and token estimation.  The system prompt is handled separately.
    pub fn prompt_text(&self) -> String {
        self.messages
            .iter()
            .filter(|m| m.role == "user")
            .map(|m| match &m.content {
                MessageContent::Text(t) => t.as_str(),
                MessageContent::Blocks(b) => b
                    .iter()
                    .filter_map(|b| b.text.as_deref())
                    .next()
                    .unwrap_or(""),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract only the most recent user message — used for semantic cache embeddings.
    /// Using the full prompt_text() in multi-turn conversations causes false semantic
    /// hits because prior turns dominate the embedding (e.g. "Thank you" after a
    /// capabilities question embeds almost identically to the original question).
    pub fn last_user_text(&self) -> String {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| match &m.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::Blocks(b) => b
                    .iter()
                    .filter_map(|b| b.text.as_deref())
                    .collect::<Vec<_>>()
                    .join(" "),
            })
            .unwrap_or_default()
    }

    /// Normalize the optional system prompt to a canonical string for cache-key
    /// computation.  Returns `None` if there is no system prompt or it is empty.
    pub fn normalized_system(&self) -> Option<String> {
        self.system.as_ref().and_then(|s| {
            let text = match s {
                serde_json::Value::String(str) => str.trim().to_string(),
                v => v.to_string(),
            };
            if text.is_empty() { None } else { Some(text) }
        })
    }

    pub fn has_tools(&self) -> bool {
        self.tools
            .as_ref()
            .and_then(|t| t.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    }

    /// Returns true if any user message contains the `![direct]` bypass annotation.
    pub fn has_direct_annotation(&self) -> bool {
        self.messages.iter().filter(|m| m.role == "user").any(|m| match &m.content {
            MessageContent::Text(t)    => t.contains("![direct]"),
            MessageContent::Blocks(bs) => bs.iter().any(|b|
                b.text.as_deref().map(|t| t.contains("![direct]")).unwrap_or(false)
            ),
        })
    }

    /// Return a clone with `![direct]` removed from all user message text so the
    /// annotation is never forwarded to the upstream model.
    pub fn strip_direct_annotation(&self) -> Self {
        let mut clone = self.clone();
        for msg in clone.messages.iter_mut().filter(|m| m.role == "user") {
            match &mut msg.content {
                MessageContent::Text(t) => {
                    *t = t.replace("![direct]", "").trim().to_string();
                }
                MessageContent::Blocks(bs) => {
                    for b in bs.iter_mut() {
                        if let Some(ref mut text) = b.text {
                            *text = text.replace("![direct]", "").trim().to_string();
                        }
                    }
                }
            }
        }
        clone
    }

    /// Check the **last** user message for an explicit quality annotation.
    /// Only checks the last message since `![good]`/`![bad]` refer to the
    /// immediately preceding response.
    pub fn extract_feedback_annotation(&self) -> Option<FeedbackSignal> {
        let text = self.messages.iter()
            .filter(|m| m.role == "user")
            .last()
            .map(|m| match &m.content {
                MessageContent::Text(t)    => t.clone(),
                MessageContent::Blocks(bs) => bs.iter()
                    .filter_map(|b| b.text.as_deref())
                    .collect::<Vec<_>>()
                    .join(" "),
            })
            .unwrap_or_default();

        if text.contains("![bad]")       { Some(FeedbackSignal::Bad)  }
        else if text.contains("![good]") { Some(FeedbackSignal::Good) }
        else { None }
    }

    /// Return a clone with **all** proxy annotations removed from every user
    /// message in one pass: `![direct]`, `![good]`, `![bad]`.
    /// Supersedes the single-annotation `strip_direct_annotation()`.
    pub fn strip_all_annotations(&self) -> Self {
        let mut clone = self.clone();
        for msg in clone.messages.iter_mut().filter(|m| m.role == "user") {
            match &mut msg.content {
                MessageContent::Text(t) => {
                    *t = t.replace("![direct]", "")
                          .replace("![good]",   "")
                          .replace("![bad]",    "")
                          .trim().to_string();
                }
                MessageContent::Blocks(bs) => {
                    for b in bs.iter_mut() {
                        if let Some(ref mut text) = b.text {
                            *text = text.replace("![direct]", "")
                                        .replace("![good]",   "")
                                        .replace("![bad]",    "")
                                        .trim().to_string();
                        }
                    }
                }
            }
        }
        clone
    }

    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// Return a clone with a speculative draft prepended to the system prompt.
    /// The draft is a cached response to a semantically similar (but not identical)
    /// question.  Sending it lets the API verify + extend rather than answer cold,
    /// reducing token usage on near-misses by roughly 1/3.
    pub fn with_draft_context(&self, draft: &str, sim_pct: u32) -> Self {
        let prefix = format!(
            "A cached response to a semantically similar question ({sim_pct}% match) is \
             provided below as a starting draft. Verify it fully answers the current question, \
             correct any inaccuracies, and extend it as needed. If it is already correct and \
             complete, return it with only minor refinements.\n\n\
             [DRAFT RESPONSE]\n{draft}\n[/DRAFT RESPONSE]"
        );
        self.with_system_prefix(&prefix)
    }

    /// Detect an implicit quality signal from conversation continuation patterns.
    ///
    /// Looks for the pattern `[prior_user, assistant, current_user]` at the end of
    /// the message list.  If the current user message contains contradiction markers
    /// the prior response was bad; affirmation markers indicate it was good.
    ///
    /// Returns `(signal, prior_user_prompt)` so the caller can classify the domain
    /// from the prior prompt, not from the contradiction phrase itself.
    pub fn detect_implicit_feedback(&self) -> Option<(FeedbackSignal, String)> {
        let msgs = &self.messages;
        if msgs.len() < 3 { return None; }

        // Need the tail to be: prior_user … assistant … current_user
        let current = msgs.last()?;
        if current.role != "user" { return None; }
        let prev = msgs.get(msgs.len() - 2)?;
        if prev.role != "assistant" { return None; }
        let prior_user = msgs.get(msgs.len() - 3)?;
        if prior_user.role != "user" { return None; }

        let text = msg_text_lower(current);
        let prior_prompt = msg_text(prior_user);

        if CONTRADICTION_MARKERS.iter().any(|m| text.contains(m)) {
            return Some((FeedbackSignal::Bad, prior_prompt));
        }
        if AFFIRMATION_MARKERS.iter().any(|m| text.contains(m)) {
            return Some((FeedbackSignal::Good, prior_prompt));
        }
        None
    }

    pub fn estimated_input_tokens(&self) -> u32 {
        (self.prompt_text().len() as u32) / 4
    }

    /// Return a clone with `prefix` prepended to the system prompt.  Used by
    /// Layer 2 to inject the distilled domain knowledge document before each
    /// local model call.
    pub fn with_system_prefix(&self, prefix: &str) -> Self {
        let mut clone = self.clone();
        clone.system = Some(match &self.system {
            Some(serde_json::Value::String(existing)) =>
                serde_json::Value::String(format!("{prefix}\n\n{existing}")),
            Some(other) =>
                serde_json::Value::String(format!("{prefix}\n\n{other}")),
            None =>
                serde_json::Value::String(prefix.to_string()),
        });
        clone
    }

    /// Return a clone with `examples` prepended as alternating user/assistant
    /// turns.  The local model sees "here's how similar questions were answered,
    /// now answer this one" — organic few-shot learning from the cache corpus.
    pub fn with_fewshot_context(&self, examples: &[FewShotExample]) -> Self {
        if examples.is_empty() { return self.clone(); }

        let mut clone    = self.clone();
        let original     = std::mem::take(&mut clone.messages);
        let mut messages = Vec::with_capacity(examples.len() * 2 + original.len());

        for ex in examples {
            messages.push(Message {
                role:    "user".to_string(),
                content: MessageContent::Text(ex.question.clone()),
            });
            messages.push(Message {
                role:    "assistant".to_string(),
                content: MessageContent::Text(ex.answer.clone()),
            });
        }
        messages.extend(original);
        clone.messages = messages;
        clone
    }
}

/// A Q&A pair drawn from the semantic cache and injected as prior conversation
/// turns before the user's actual query.  The `sim` field records how close
/// this example was to the current prompt — useful for future trace mode.
#[derive(Debug, Clone)]
pub struct FewShotExample {
    pub question: String,
    pub answer:   String,
    pub sim:      f64,
}

/// Explicit quality signal from a `![good]` or `![bad]` annotation in the
/// last user message.  Stripped before forwarding; recorded in the feedback
/// table so the adaptive threshold adaptor can factor user satisfaction into
/// the per-domain routing calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeedbackSignal {
    Good,
    Bad,
}

impl FeedbackSignal {
    pub fn as_str(self) -> &'static str {
        match self { FeedbackSignal::Good => "good", FeedbackSignal::Bad => "bad" }
    }
}

// ── Implicit feedback detection ────────────────────────────────────────────

static CONTRADICTION_MARKERS: &[&str] = &[
    "that's wrong", "thats wrong", "that is wrong",
    "you're wrong", "youre wrong",
    "not correct", "incorrect",
    "not what i meant", "not what i asked", "not what i need",
    "that doesn't work", "that didnt work", "that didn't work",
    "doesn't compile", "didnt compile", "didn't compile",
    "that failed", "that errors", "that throws",
    "actually no", "no that's", "no, that", "no, it",
    "not quite right", "that's not right", "that's not what",
    "wrong answer", "wrong approach",
];

static AFFIRMATION_MARKERS: &[&str] = &[
    "thank you", "thanks!", "thanks,",
    "perfect", "exactly what i", "exactly right",
    "that worked", "that works", "works great", "works perfectly",
    "that's exactly", "that's perfect", "that's correct",
    "great answer", "great, now", "great! now",
    "well done",
];

fn msg_text(m: &Message) -> String {
    match &m.content {
        MessageContent::Text(t)    => t.clone(),
        MessageContent::Blocks(bs) => bs.iter()
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn msg_text_lower(m: &Message) -> String {
    msg_text(m).to_lowercase()
}

impl MessagesResponse {
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter(|b| b.kind == "text")
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns true if the response body contains any tool_use blocks, meaning
    /// the model invoked a tool.  Distinct from whether the *request* offered
    /// tools — a text answer to a tool-equipped request is still cacheable.
    pub fn uses_tools(&self) -> bool {
        self.content.iter().any(|b| b.kind == "tool_use")
    }
}

// ── Backend trait ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BackendResult {
    pub response:   MessagesResponse,
    pub confidence: Option<f64>,
    pub latency_ms: u64,
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn complete(&self, req: &MessagesRequest) -> Result<BackendResult>;
    fn name(&self) -> &'static str;

    /// Called before routing to check if the backend is reachable.
    /// Defaults to true — individual backends may override with a health ping.
    async fn is_available(&self) -> bool {
        true
    }
}
