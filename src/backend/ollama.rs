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
struct OllamaRequest {
    model:    String,
    messages: Vec<OllamaMessage>,
    stream:   bool,
    options:  OllamaOptions,
    // "json" string works on all Ollama versions; JSON schema objects require 0.4+.
    #[serde(skip_serializing_if = "Option::is_none")]
    format:   Option<String>,
}

#[derive(Serialize)]
struct OllamaMessage {
    role:    String,
    content: String,
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

/// Parsed response from the structured-output wrapper.
#[derive(Deserialize)]
struct StructuredResponse {
    answer:     String,
    confidence: f64,
}

// ── System instruction for structured output ───────────────────────────────

const CONFIDENCE_SYSTEM: &str =
    "Respond ONLY with valid JSON matching this exact schema — no markdown, no extra text: \
     {\"answer\": \"<your full response here>\", \"confidence\": <decimal 0.0-1.0>}. \
     Set confidence to how confident you are in the correctness and completeness of the answer.";

// ── Backend impl ──────────────────────────────────────────────────────────

#[async_trait]
impl ModelBackend for OllamaBackend {
    async fn complete(&self, req: &MessagesRequest) -> Result<BackendResult> {
        let start = Instant::now();
        let url   = format!("{}/api/chat", self.base_url);

        // Build system message: merge confidence instruction with any user-provided system prompt.
        // Use as_str() to extract the raw string value — Display on Value::String includes
        // JSON double-quotes which would be injected verbatim into the Ollama system content.
        let sys_content = match &req.system {
            Some(serde_json::Value::String(s)) => format!("{CONFIDENCE_SYSTEM}\n\n{s}"),
            Some(_) | None                     => CONFIDENCE_SYSTEM.to_string(),
        };

        let mut messages = Vec::with_capacity(req.messages.len() + 1);
        messages.push(OllamaMessage { role: "system".into(), content: sys_content });

        for m in &req.messages {
            let content = match &m.content {
                super::MessageContent::Text(t)    => t.clone(),
                super::MessageContent::Blocks(bs) => bs
                    .iter()
                    .filter_map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            messages.push(OllamaMessage { role: m.role.clone(), content });
        }

        let body = OllamaRequest {
            model:    self.model_id.clone(),
            messages,
            stream:   false,
            options:  OllamaOptions { temperature: 0.1 },
            format:   Some("json".to_string()),
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
        let raw        = ollama.message.content.clone();

        // Parse the structured response.  Fall back to the heuristic if the
        // model ignored the format instruction.
        let (answer_text, confidence) =
            if let Ok(s) = serde_json::from_str::<StructuredResponse>(&raw) {
                (s.answer, s.confidence.clamp(0.0, 1.0))
            } else {
                let conf = estimate_confidence(&raw, req.estimated_input_tokens());
                (raw, conf)
            };

        let response = MessagesResponse {
            id:          uuid::Uuid::new_v4().to_string(),
            kind:        "message".into(),
            role:        "assistant".into(),
            content:     vec![ContentBlock {
                kind:  "text".into(),
                text:  Some(answer_text),
                extra: Default::default(),
            }],
            model:       self.model_id.clone(),
            stop_reason: Some("end_turn".into()),
            usage:       Usage {
                input_tokens:  ollama.prompt_eval_count.max(req.estimated_input_tokens()),
                output_tokens: ollama.eval_count,
            },
        };

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

/// Fallback confidence heuristic used when the model does not return structured output.
fn estimate_confidence(response_text: &str, input_tokens: u32) -> f64 {
    let resp_len  = response_text.len();
    let has_code  = response_text.contains("```");
    let has_punct = response_text.contains('.') || response_text.contains('\n');

    if resp_len < 20 {
        return 0.40;
    }

    let ratio = resp_len as f64 / (input_tokens as f64 * 4.0 + 1.0);
    let base   = (0.50 + ratio * 0.15).min(0.85);

    let bonus = if has_code  { 0.05 } else { 0.0 }
              + if has_punct { 0.03 } else { 0.0 };

    (base + bonus).min(1.0)
}
