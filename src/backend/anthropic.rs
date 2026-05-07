use anyhow::{bail, Result};
use async_trait::async_trait;
use std::time::Instant;

use crate::auth::Credentials;
use crate::config::ApiConfig;
use super::{BackendResult, MessagesRequest, MessagesResponse, ModelBackend};

pub struct AnthropicBackend {
    client:   reqwest::Client,
    base_url: String,
    creds:    Credentials,
    model:    String,
}

impl AnthropicBackend {
    pub fn new(cfg: &ApiConfig, creds: Credentials) -> Self {
        AnthropicBackend {
            client: reqwest::Client::builder()
                .use_rustls_tls()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("reqwest client"),
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            creds,
            model: cfg.model.clone(),
        }
    }

    fn auth_header(&self) -> String {
        // OAuth tokens from claude desktop start with "sk-ant-oat-"
        if self.creds.api_key.starts_with("sk-ant-oat") {
            format!("Bearer {}", self.creds.api_key)
        } else {
            self.creds.api_key.clone()
        }
    }

    fn auth_header_name(&self) -> &'static str {
        if self.creds.api_key.starts_with("sk-ant-oat") {
            "Authorization"
        } else {
            "x-api-key"
        }
    }
}

#[async_trait]
impl ModelBackend for AnthropicBackend {
    async fn complete(&self, req: &MessagesRequest) -> Result<BackendResult> {
        let start = Instant::now();
        let url   = format!("{}/v1/messages", self.base_url);

        // Override model with configured default if the client sent a placeholder
        let mut body = req.clone();
        body.stream  = Some(false); // non-streaming path
        body.model   = self.model.clone();

        let resp = self.client
            .post(&url)
            .header(self.auth_header_name(), self.auth_header())
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text   = resp.text().await.unwrap_or_default();
            bail!("Anthropic API {status}: {text}");
        }

        let response: MessagesResponse = resp.json().await?;
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(BackendResult {
            response,
            confidence: None, // API responses are authoritative
            latency_ms,
        })
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}

// ── Streaming passthrough ──────────────────────────────────────────────────
// When the client requests streaming we forward the raw SSE bytes and
// simultaneously accumulate the full response text for caching.

use bytes::Bytes;
use futures_util::StreamExt;
use tokio::sync::mpsc;

pub struct StreamAccumulator {
    pub text:         String,
    /// Real Anthropic message ID from the `message_start` SSE event (e.g. "msg_01abc…").
    pub message_id:   String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl AnthropicBackend {
    /// Forward a streaming request to Anthropic, tee-ing the raw SSE to `tx`
    /// while accumulating content for caching.  Returns accumulated text + usage.
    pub async fn stream_to_channel(
        &self,
        req: &MessagesRequest,
        tx: mpsc::Sender<Result<Bytes, std::io::Error>>,
    ) -> Result<StreamAccumulator> {
        let url  = format!("{}/v1/messages", self.base_url);
        let mut body = req.clone();
        body.stream  = Some(true);
        body.model   = self.model.clone();

        let resp = self.client
            .post(&url)
            .header(self.auth_header_name(), self.auth_header())
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text   = resp.text().await.unwrap_or_default();
            bail!("Anthropic stream {status}: {text}");
        }

        let mut stream = resp.bytes_stream();
        let mut acc = StreamAccumulator {
            text:          String::new(),
            message_id:    String::new(),
            input_tokens:  req.estimated_input_tokens(), // overwritten by message_start event
            output_tokens: 0,
        };

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            acc.accumulate_sse_chunk(&chunk);
            let _ = tx.send(Ok(chunk)).await;
        }

        Ok(acc)
    }
}

impl StreamAccumulator {
    fn accumulate_sse_chunk(&mut self, chunk: &[u8]) {
        if let Ok(text) = std::str::from_utf8(chunk) {
            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                        match v["type"].as_str() {
                            // message_start: real message ID + actual input token count
                            Some("message_start") => {
                                if let Some(id) = v["message"]["id"].as_str() {
                                    self.message_id = id.to_string();
                                }
                                if let Some(n) = v["message"]["usage"]["input_tokens"].as_u64() {
                                    self.input_tokens = n as u32;
                                }
                            }
                            // content_block_delta: accumulate text
                            Some("content_block_delta") => {
                                if let Some(t) = v["delta"]["text"].as_str() {
                                    self.text.push_str(t);
                                    self.output_tokens += (t.len() as u32) / 4;
                                }
                            }
                            // message_delta: replace running estimate with final output count
                            Some("message_delta") => {
                                if let Some(u) = v["usage"]["output_tokens"].as_u64() {
                                    self.output_tokens = u as u32;
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
