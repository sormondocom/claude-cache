use anyhow::{bail, Result};
use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::auth::{Credentials, CredentialStore};
use crate::config::ApiConfig;
use crate::error::ProxyError;
use super::{BackendResult, MessagesRequest, MessagesResponse, ModelBackend};

pub struct AnthropicBackend {
    client:         reqwest::Client,
    base_url:       String,
    creds:          CredentialStore,
    model:          String,
    max_retries:    u32,
    retry_delay_ms: u64,
    last_reload:    AtomicU64,
}

// ── Diagnostics ────────────────────────────────────────────────────────────

/// 429 with no x-ratelimit-* headers means the token has no direct API access.
/// Claude Pro OAuth tokens typically trigger this; Claude Max subscribers may or
/// may not depending on plan tier.  This is permanent — retrying will not help.
pub fn is_no_api_access(status: reqwest::StatusCode, headers: &reqwest::header::HeaderMap) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS
        && headers.get("x-ratelimit-remaining-requests").is_none()
        && headers.get("x-ratelimit-limit-requests").is_none()
}

/// True when the Anthropic API 400 body indicates the proxy's credit balance is zero.
fn is_credit_exhausted_body(text: &str) -> bool {
    let m = text.to_lowercase();
    (m.contains("credit") && m.contains("too low"))
        || m.contains("insufficient_credits")
        || m.contains("credit balance is too low")
}

/// Log every non-success Anthropic response in full — status, all response
/// headers, and the complete body.  No filtering or special-casing.
fn log_api_error(
    status:  reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body:    &str,
    label:   &str,
) {
    // Collect every header into a readable k: v list.
    let hdrs: Vec<String> = headers
        .iter()
        .filter_map(|(k, v)| v.to_str().ok().map(|v| format!("{}: {}", k, v)))
        .collect();
    let hdrs_str = if hdrs.is_empty() { "(none)".to_owned() } else { hdrs.join(", ") };

    if is_no_api_access(status, headers) {
        tracing::warn!(
            "{label} 429 — token has no direct API access (no x-ratelimit headers). \
             Claude Pro subscriptions typically cannot call api.anthropic.com; \
             Max subscriptions may have API access depending on the plan tier. \
             Add ANTHROPIC_API_KEY (console.anthropic.com) for guaranteed API access.\n  body: {body}"
        );
    } else {
        tracing::warn!(
            "{label} error\n  status:  {status}\n  headers: {hdrs_str}\n  body:    {body}"
        );
    }
}

impl AnthropicBackend {
    pub fn new(cfg: &ApiConfig, creds: CredentialStore) -> Self {
        AnthropicBackend {
            client: reqwest::Client::builder()
                .use_rustls_tls()
                .timeout(std::time::Duration::from_secs(cfg.request_timeout_secs))
                .build()
                .expect("reqwest client"),
            base_url:       cfg.base_url.trim_end_matches('/').to_string(),
            creds,
            model:          cfg.model.clone(),
            max_retries:    cfg.max_retries,
            retry_delay_ms: cfg.retry_delay_ms,
            last_reload:    AtomicU64::new(0),
        }
    }

    /// Reload credentials with a 30-second debounce to prevent concurrent 401
    /// errors from all triggering reloads simultaneously.
    fn try_reload_creds(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last = self.last_reload.load(Ordering::Relaxed);
        if now.saturating_sub(last) < 30 {
            return false; // another request already reloaded recently
        }
        self.last_reload.store(now, Ordering::Relaxed);
        self.creds.reload().is_ok()
    }

    /// Returns the correct header name + value for the current credentials.
    /// OAuth tokens (sk-ant-oat…) use Bearer; plain API keys use x-api-key.
    fn auth_parts(creds: &Credentials) -> (&'static str, String) {
        if creds.api_key.starts_with("sk-ant-oat") {
            ("Authorization", format!("Bearer {}", creds.api_key))
        } else {
            ("x-api-key", creds.api_key.clone())
        }
    }

    /// Send one request to Anthropic, retrying once on 401 (OAuth rotation).
    /// Returns the raw response — caller checks status and handles the body.
    async fn send_once(&self, body: &MessagesRequest, url: &str) -> Result<reqwest::Response> {
        for attempt in 0..2u8 {
            let creds = self.creds.get();
            let (auth_name, auth_val) = Self::auth_parts(&creds);
            let mut builder = self.client
                .post(url)
                .header(auth_name, auth_val)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json");
            // OAuth (Bearer) requests must identify as claude-code so Anthropic's
            // API routes them through the subscription path, not the API-key path.
            if auth_name == "Authorization" {
                builder = builder.header("user-agent", "claude-code/1.0.0");
            }
            if let Some(beta) = &body.anthropic_beta {
                builder = builder.header("anthropic-beta", beta.as_str());
            }
            let resp = builder.json(body).send().await?;
            if resp.status() == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                if self.try_reload_creds() {
                    tracing::info!("auth: OAuth token rotated, retrying");
                    continue;
                }
            }
            return Ok(resp);
        }
        bail!("Anthropic API: authentication failed after credential reload")
    }

    /// Exponential backoff delay for retry `n` (1-based): base * 2^(n-1), capped at 8× base.
    fn retry_delay(&self, n: u32) -> std::time::Duration {
        let factor = 1u64 << (n - 1).min(3);
        std::time::Duration::from_millis(self.retry_delay_ms.saturating_mul(factor))
    }
}

#[async_trait]
impl ModelBackend for AnthropicBackend {
    async fn complete(&self, req: &MessagesRequest) -> Result<BackendResult> {
        let start = Instant::now();
        let url   = format!("{}/v1/messages", self.base_url);

        let mut body = req.clone();
        body.stream  = Some(false);
        body.model   = self.model.clone();

        let mut last_err = anyhow::anyhow!("no attempts made");
        for retry in 0..=self.max_retries {
            if retry > 0 {
                let delay = self.retry_delay(retry);
                tracing::warn!("Anthropic retry {retry}/{} after {}ms",
                    self.max_retries, delay.as_millis());
                tokio::time::sleep(delay).await;
            }

            let resp = self.send_once(&body, &url).await?;

            if !resp.status().is_success() {
                let status  = resp.status();
                let headers = resp.headers().clone();
                let text    = resp.text().await.unwrap_or_default();
                log_api_error(status, &headers, &text, "Anthropic API");
                if is_no_api_access(status, &headers) {
                    return Err(ProxyError::NoApiAccess(text).into());
                }
                if is_credit_exhausted_body(&text) {
                    return Err(ProxyError::CreditExhausted(text).into());
                }
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    let err: anyhow::Error = ProxyError::RateLimited(text).into();
                    if retry < self.max_retries {
                        last_err = err;
                        // Genuine subscription rate limit — Anthropic sends retry-after.
                        // Pro subscribers hit per-minute limits often; respect the header.
                        if headers.get("x-ratelimit-limit-requests").is_some() {
                            let wait = headers.get("retry-after")
                                .and_then(|v| v.to_str().ok())
                                .and_then(|s| s.parse::<u64>().ok())
                                .unwrap_or(5)
                                .min(60);
                            tracing::warn!("rate limited — waiting {wait}s (retry {}/{})",
                                retry + 1, self.max_retries);
                            tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                        }
                        // overloaded_error: no sleep here; exponential backoff fires at top of loop
                        continue;
                    }
                    return Err(err);
                }
                return Err(anyhow::anyhow!("Anthropic API {status}: {text}"));
            }

            let response: MessagesResponse = resp.json().await?;
            return Ok(BackendResult {
                response,
                confidence: None,
                latency_ms: start.elapsed().as_millis() as u64,
            });
        }

        Err(last_err)
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}

// ── Streaming passthrough ──────────────────────────────────────────────────

use bytes::Bytes;
use futures_util::StreamExt;
use tokio::sync::mpsc;

pub struct StreamAccumulator {
    pub text:          String,
    /// Real Anthropic message ID from the `message_start` SSE event.
    pub message_id:    String,
    pub input_tokens:  u32,
    pub output_tokens: u32,
    /// Actual stop reason from the `message_delta` SSE event.
    pub stop_reason:   String,
    /// True if the stream contained at least one `tool_use` content block.
    /// When true the accumulated `text` is incomplete and should not be cached.
    pub has_tool_use:  bool,
}

impl AnthropicBackend {
    /// Forward a streaming request to Anthropic, tee-ing raw SSE to `tx`
    /// while accumulating content for caching.  Retries on 401 (via send_once)
    /// and on overloaded_error (up to max_retries times, with backoff).
    pub async fn stream_to_channel(
        &self,
        req: &MessagesRequest,
        tx:  mpsc::Sender<Result<Bytes, std::io::Error>>,
    ) -> Result<StreamAccumulator> {
        let url  = format!("{}/v1/messages", self.base_url);
        let mut body = req.clone();
        body.stream  = Some(true);
        body.model   = self.model.clone();

        let mut last_err = anyhow::anyhow!("no attempts made");
        for retry in 0..=self.max_retries {
            if retry > 0 {
                let delay = self.retry_delay(retry);
                tracing::warn!("Anthropic stream retry {retry}/{} after {}ms",
                    self.max_retries, delay.as_millis());
                tokio::time::sleep(delay).await;
            }

            let resp = self.send_once(&body, &url).await?;

            if !resp.status().is_success() {
                let status  = resp.status();
                let headers = resp.headers().clone();
                let text    = resp.text().await.unwrap_or_default();
                log_api_error(status, &headers, &text, "Anthropic stream");
                if is_no_api_access(status, &headers) {
                    bail!("Anthropic API no_api_access: {text}");
                }
                let err = anyhow::anyhow!("Anthropic stream {status}: {text}");
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS && retry < self.max_retries {
                    last_err = err;
                    if headers.get("x-ratelimit-limit-requests").is_some() {
                        let wait = headers.get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse::<u64>().ok())
                            .unwrap_or(5)
                            .min(60);
                        tracing::warn!("stream rate limited — waiting {wait}s (retry {}/{})",
                            retry + 1, self.max_retries);
                        tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    }
                    continue;
                }
                return Err(err);
            }

            let mut stream = resp.bytes_stream();
            let mut acc = StreamAccumulator {
                text:          String::new(),
                message_id:    String::new(),
                input_tokens:  req.estimated_input_tokens(),
                output_tokens: 0,
                stop_reason:   "end_turn".to_string(),
                has_tool_use:  false,
            };

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                acc.accumulate_sse_chunk(&chunk);
                let _ = tx.send(Ok(chunk)).await;
            }

            return Ok(acc);
        }

        Err(last_err)
    }
}

impl StreamAccumulator {
    fn accumulate_sse_chunk(&mut self, chunk: &[u8]) {
        if let Ok(text) = std::str::from_utf8(chunk) {
            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                        match v["type"].as_str() {
                            Some("message_start") => {
                                if let Some(id) = v["message"]["id"].as_str() {
                                    self.message_id = id.to_string();
                                }
                                if let Some(n) = v["message"]["usage"]["input_tokens"].as_u64() {
                                    self.input_tokens = n as u32;
                                }
                            }
                            Some("content_block_start") => {
                                if v["content_block"]["type"].as_str() == Some("tool_use") {
                                    self.has_tool_use = true;
                                }
                            }
                            Some("content_block_delta") => {
                                if let Some(t) = v["delta"]["text"].as_str() {
                                    self.text.push_str(t);
                                    self.output_tokens = self.output_tokens.saturating_add((t.len() as u32) / 4);
                                }
                            }
                            Some("message_delta") => {
                                if let Some(u) = v["usage"]["output_tokens"].as_u64() {
                                    self.output_tokens = u as u32;
                                }
                                if let Some(r) = v["delta"]["stop_reason"].as_str() {
                                    self.stop_reason = r.to_string();
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}
