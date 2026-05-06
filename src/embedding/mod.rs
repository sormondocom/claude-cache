use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;

use crate::config::EmbeddingConfig;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn model(&self) -> &str;
    fn dimensions(&self) -> usize;
}

// ── Ollama embedder ────────────────────────────────────────────────────────

pub struct OllamaEmbedder {
    client:  reqwest::Client,
    base_url: String,
    model:   String,
    dims:    usize,
}

impl OllamaEmbedder {
    pub fn new(cfg: &EmbeddingConfig) -> Self {
        OllamaEmbedder {
            client:   reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("reqwest client"),
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model:    cfg.model.clone(),
            dims:     cfg.dimensions,
        }
    }
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model:  &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let body = EmbedRequest { model: &self.model, prompt: text };
        let resp: EmbedResponse = self.client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp.embedding)
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

// ── Stub (when embedding disabled / Ollama unavailable) ───────────────────

pub struct StubEmbedder {
    dims: usize,
}

impl StubEmbedder {
    pub fn new(dims: usize) -> Self {
        StubEmbedder { dims }
    }
}

#[async_trait]
impl Embedder for StubEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Deterministic word-hash bag-of-words — NOT semantic, only for fallback
        let mut v = vec![0.0f32; self.dims];
        for (i, word) in text.split_whitespace().enumerate() {
            let h = fnv1a(word);
            let idx = (h as usize) % self.dims;
            v[idx] += 1.0 / (i as f32 + 1.0);
        }
        Ok(normalize_l2(v))
    }

    fn model(&self) -> &str {
        "stub"
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

fn fnv1a(s: &str) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    s.bytes().fold(OFFSET, |h, b| (h ^ b as u64).wrapping_mul(PRIME))
}

fn normalize_l2(mut v: Vec<f32>) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        v.iter_mut().for_each(|x| *x /= mag);
    }
    v
}
