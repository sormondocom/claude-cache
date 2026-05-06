use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::config::LocalConfig;
use super::{
    BackendResult, ContentBlock, MessagesRequest, MessagesResponse, ModelBackend, Usage,
};

pub struct OllamaBackend {
    client:   reqwest::Client,
    base_url: String,
    model_id: String,
}

impl OllamaBackend {
    pub fn new(cfg: &LocalConfig) -> Self {
        OllamaBackend {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(cfg.timeout_secs))
                .build()
                .expect("reqwest client"),
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model_id: cfg.model_id.clone(),
        }
    }
}

// ── Ollama wire types ──────────────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model:    &'a str,
    messages: Vec<OllamaMessage<'a>>,
    stream:   bool,
    options:  OllamaOptions,
}

#[derive(Serialize)]
struct OllamaMessage<'a> {
    role:    &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: OllamaMessageOut,
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    prompt_eval_count: u32,
}

#[derive(Deserialize)]
struct OllamaMessageOut {
    content: String,
}

#[async_trait]
impl ModelBackend for OllamaBackend {
    async fn complete(&self, req: &MessagesRequest) -> Result<BackendResult> {
        let start = Instant::now();
        let url   = format!("{}/api/chat", self.base_url);

        let messages: Vec<OllamaMessage> = req
            .messages
            .iter()
            .map(|m| {
                let content = match &m.content {
                    super::MessageContent::Text(t) => t.as_str(),
                    super::MessageContent::Blocks(b) => b
                        .iter()
                        .filter_map(|b| b.text.as_deref())
                        .next()
                        .unwrap_or(""),
                };
                OllamaMessage { role: &m.role, content }
            })
            .collect();

        let body = OllamaRequest {
            model:    &self.model_id,
            messages,
            stream:   false,
            options:  OllamaOptions { temperature: 0.1 },
        };

        let http_resp = self.client
            .post(&url)
            .json(&body)
            .send()
            .await?;

        if !http_resp.status().is_success() {
            let status = http_resp.status();
            let text   = http_resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama {status}: {text}");
        }

        let ollama: OllamaResponse = http_resp.json().await?;
        let latency_ms = start.elapsed().as_millis() as u64;

        let response = MessagesResponse {
            id:          uuid::Uuid::new_v4().to_string(),
            kind:        "message".into(),
            role:        "assistant".into(),
            content:     vec![ContentBlock {
                kind: "text".into(),
                text: Some(ollama.message.content.clone()),
            }],
            model:       self.model_id.clone(),
            stop_reason: Some("end_turn".into()),
            usage:       Usage {
                input_tokens:  ollama.prompt_eval_count.max(req.estimated_input_tokens()),
                output_tokens: ollama.eval_count,
            },
        };

        // Confidence heuristic based on response length and latency.
        // A very short response to a complex question is low-confidence.
        let confidence = estimate_confidence(&ollama.message.content, req.estimated_input_tokens());

        Ok(BackendResult { response, confidence: Some(confidence), latency_ms })
    }

    fn name(&self) -> &'static str {
        "ollama"
    }

    async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        self.client
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

/// Rough confidence score for a local model response.
/// Longer, well-structured responses to short prompts = higher confidence.
/// Very terse responses to long prompts = lower confidence.
fn estimate_confidence(response_text: &str, input_tokens: u32) -> f64 {
    let resp_len  = response_text.len();
    let has_code  = response_text.contains("```");
    let has_punct = response_text.contains('.') || response_text.contains('\n');

    // Very short response is suspect
    if resp_len < 20 {
        return 0.40;
    }

    let ratio = resp_len as f64 / (input_tokens as f64 * 4.0 + 1.0);
    let base   = (0.50 + ratio * 0.15).min(0.85);

    let bonus = if has_code  { 0.05 } else { 0.0 }
              + if has_punct { 0.03 } else { 0.0 };

    (base + bonus).min(1.0)
}
