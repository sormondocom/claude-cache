pub mod anthropic;
pub mod ollama;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub use anthropic::AnthropicBackend;
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
    /// Extract the concatenated text of all user messages — used for cache keying.
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

    pub fn has_tools(&self) -> bool {
        self.tools
            .as_ref()
            .and_then(|t| t.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    }

    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    pub fn estimated_input_tokens(&self) -> u32 {
        (self.prompt_text().len() as u32) / 4
    }
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
