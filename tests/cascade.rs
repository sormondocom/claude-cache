//! Integration tests for the routing cascade.
//!
//! Each test uses real async SQLite stores (tempfile) and mock ModelBackend
//! implementations so routing decisions are deterministic without network calls.

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use tempfile::tempdir;

use claude_cache::{
    backend::{BackendResult, ContentBlock, Message, MessageContent, MessagesRequest, MessagesResponse, ModelBackend, Usage},
    budget::BudgetLedger,
    cache::CacheStore,
    config::{AppConfig, BudgetConfig, RoutingConfig},
    domain::ShapeKey,
    embedding::StubEmbedder,
    identity::NodeIdentity,
    router::{RouteDecision, Router},
};

// ── Mock backends ─────────────────────────────────────────────────────────────

struct MockBackend {
    name_str:   &'static str,
    confidence: Option<f64>,
    text:       &'static str,
}

impl MockBackend {
    fn api(text: &'static str) -> Self {
        MockBackend { name_str: "mock-api", confidence: None, text }
    }
    fn local(text: &'static str, confidence: f64) -> Self {
        MockBackend { name_str: "mock-local", confidence: Some(confidence), text }
    }
}

#[async_trait]
impl ModelBackend for MockBackend {
    async fn complete(&self, req: &MessagesRequest) -> Result<BackendResult> {
        Ok(BackendResult {
            response: MessagesResponse {
                id:          "test-id".into(),
                kind:        "message".into(),
                role:        "assistant".into(),
                content:     vec![ContentBlock { kind: "text".into(), text: Some(self.text.into()) }],
                model:       self.name_str.into(),
                stop_reason: Some("end_turn".into()),
                usage:       Usage { input_tokens: 10, output_tokens: 20 },
            },
            confidence: self.confidence,
            latency_ms: 1,
        })
    }
    fn name(&self) -> &'static str { self.name_str }
}

// ── Test environment helpers ──────────────────────────────────────────────────

struct Env {
    _dir:   tempfile::TempDir,
    cache:  Arc<CacheStore>,
    budget: Arc<BudgetLedger>,
    cfg:    Arc<AppConfig>,
    id:     Arc<NodeIdentity>,
}

impl Env {
    async fn new() -> Self {
        let dir  = tempdir().unwrap();
        let id   = Arc::new(NodeIdentity::generate());

        let cache_path  = dir.path().join("cache.db").to_str().unwrap().to_string();
        let budget_path = dir.path().join("budget.db").to_str().unwrap().to_string();

        let cache  = Arc::new(CacheStore::open(&cache_path, &id.fingerprint).await.unwrap());
        let bcfg   = BudgetConfig {
            db_path:           budget_path,
            daily_limit_usd:   10.0,
            warn_at_pct:       80,
            input_per_1k_usd:  0.003,
            output_per_1k_usd: 0.015,
        };
        let budget = Arc::new(BudgetLedger::open(bcfg).await.unwrap());
        let cfg    = Arc::new(AppConfig::default());

        Env { _dir: dir, cache, budget, cfg, id }
    }

    /// Build a Router that always routes locally (thresholds all 1.0).
    fn router_force_local(&self, local_confidence: f64) -> Router {
        let mut cfg = (*self.cfg).clone();
        cfg.routing = RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0 };
        cfg.local.confidence_floor = local_confidence;
        cfg.local.enabled = true;
        cfg.embedding.enabled = false; // stub only in tests

        Router::new(
            Arc::new(cfg),
            self.cache.clone(),
            self.budget.clone(),
            Arc::new(StubEmbedder::new(64)),
            Arc::new(MockBackend::local("local answer", 0.90)),
            Arc::new(MockBackend::api("api answer")),
        )
    }

    /// Build a Router that always routes to API (thresholds all 0.0).
    fn router_force_api(&self) -> Router {
        let mut cfg = (*self.cfg).clone();
        cfg.routing = RoutingConfig { novelty_threshold: 0.0, complexity_threshold: 0.0, consequence_threshold: 0.0 };
        cfg.local.enabled = true;
        cfg.embedding.enabled = false;

        Router::new(
            Arc::new(cfg),
            self.cache.clone(),
            self.budget.clone(),
            Arc::new(StubEmbedder::new(64)),
            Arc::new(MockBackend::local("local answer", 0.90)),
            Arc::new(MockBackend::api("api answer")),
        )
    }

    /// Custom router with explicit backends.
    fn router_with(
        &self,
        routing: RoutingConfig,
        confidence_floor: f64,
        local: impl ModelBackend + 'static,
        api: impl ModelBackend + 'static,
    ) -> Router {
        let mut cfg = (*self.cfg).clone();
        cfg.routing = routing;
        cfg.local.confidence_floor = confidence_floor;
        cfg.local.enabled = true;
        cfg.embedding.enabled = false;
        Router::new(
            Arc::new(cfg),
            self.cache.clone(),
            self.budget.clone(),
            Arc::new(StubEmbedder::new(64)),
            Arc::new(local),
            Arc::new(api),
        )
    }
}

fn user_req(text: &str) -> MessagesRequest {
    MessagesRequest {
        model:      "claude-sonnet-4-6".into(),
        messages:   vec![Message { role: "user".into(), content: MessageContent::Text(text.into()) }],
        max_tokens: 1024,
        system:     None,
        stream:     None,
        tools:      None,
        extra:      Default::default(),
    }
}

// ── Cascade tests ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn exact_cache_hit_skips_backends() {
    let env    = Env::new().await;
    let router = env.router_force_api(); // would go to API on miss

    let req    = user_req("what is a Rust lifetime?");

    // First call — cache miss, goes to API
    let r1 = router.route(&req).await.unwrap();
    assert_eq!(r1.decision, RouteDecision::Api);

    // Second call — must be a cache hit
    let r2 = router.route(&req).await.unwrap();
    assert_eq!(r2.decision, RouteDecision::ExactCache);
    assert_eq!(r2.response.text_content(), "api answer");
}

#[tokio::test]
async fn local_model_high_confidence_serves_locally() {
    let env    = Env::new().await;
    let router = env.router_force_local(0.80); // floor=0.80, mock returns 0.90

    let req = user_req("explain what a Rust closure is");
    let r   = router.route(&req).await.unwrap();
    assert_eq!(r.decision, RouteDecision::LocalModel);
    assert_eq!(r.response.text_content(), "local answer");
}

#[tokio::test]
async fn local_model_low_confidence_escalates_to_api() {
    let env = Env::new().await;
    // Force routing gate open (always try local), but local returns confidence 0.5 < floor 0.8
    let router = env.router_with(
        RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0 },
        0.80,
        MockBackend::local("weak answer", 0.50), // below floor
        MockBackend::api("strong api answer"),
    );

    let req = user_req("implement a lock-free queue in Rust");
    let r   = router.route(&req).await.unwrap();
    assert_eq!(r.decision, RouteDecision::Api);
    assert_eq!(r.response.text_content(), "strong api answer");
}

#[tokio::test]
async fn tool_use_always_goes_to_api() {
    let env    = Env::new().await;
    let router = env.router_force_local(0.0); // would normally go local

    let mut req = user_req("call my tool");
    req.tools   = Some(serde_json::json!([{ "name": "my_tool", "description": "does stuff" }]));

    let r = router.route(&req).await.unwrap();
    assert_eq!(r.decision, RouteDecision::Api, "tool-use must bypass cache and local model");
}

#[tokio::test]
async fn recency_bypass_skips_cache() {
    let env    = Env::new().await;
    let router = env.router_force_api();

    // Seed the cache with a "stable" version of this prompt
    let stable_req = user_req("what is the latest version of tokio");
    let r1 = router.route(&stable_req).await.unwrap();
    assert_eq!(r1.decision, RouteDecision::Api);

    // A "latest / new" query should bypass the cache (policy recency bypass)
    let recency_req = user_req("what is new in the latest tokio release today");
    let r2 = router.route(&recency_req).await.unwrap();
    // Policy forces bypass → goes to API again, NOT cache
    assert_eq!(r2.decision, RouteDecision::Api);
}

#[tokio::test]
async fn cache_hit_returns_saved_response() {
    let env    = Env::new().await;

    // Manually seed the cache
    let shape = ShapeKey { domain: "rust".into(), intent: "explain".into(), complexity: 0.3 };
    let prompt = "what is ownership in Rust";
    let stored_resp = serde_json::to_string(&MessagesResponse {
        id:          "cached-id".into(),
        kind:        "message".into(),
        role:        "assistant".into(),
        content:     vec![ContentBlock { kind: "text".into(), text: Some("ownership means one owner".into()) }],
        model:       "anthropic".into(),
        stop_reason: Some("end_turn".into()),
        usage:       Usage { input_tokens: 5, output_tokens: 10 },
    }).unwrap();

    env.cache.store(&shape, prompt, &stored_resp, "anthropic", None, Some(3600), false).await.unwrap();

    // Route the same prompt — must hit cache
    let router = env.router_force_api();
    let req    = user_req(prompt);
    let r      = router.route(&req).await.unwrap();

    assert_eq!(r.decision, RouteDecision::ExactCache);
    assert_eq!(r.response.text_content(), "ownership means one owner");
}

#[tokio::test]
async fn budget_ceiling_forces_local_over_api() {
    let env = Env::new().await;

    // Set a zero budget so the first API call exhausts it
    let mut cfg = (*env.cfg).clone();
    cfg.routing = RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0 };
    cfg.local.confidence_floor = 0.0; // accept any local confidence
    cfg.local.enabled = true;
    cfg.embedding.enabled = false;
    cfg.budget.daily_limit_usd = 0.0; // already exceeded

    let router = Router::new(
        Arc::new(cfg),
        env.cache.clone(),
        env.budget.clone(),
        Arc::new(StubEmbedder::new(64)),
        Arc::new(MockBackend::local("local fallback", 0.50)),
        Arc::new(MockBackend::api("api answer")),
    );

    let req = user_req("write me a web server in Rust");
    let r   = router.route(&req).await.unwrap();
    // Budget exceeded → forced to local regardless of routing gate
    assert_eq!(r.decision, RouteDecision::LocalModel);
}

#[tokio::test]
async fn routing_gate_high_consequence_forces_api() {
    let env    = Env::new().await;
    // Use default thresholds — "review" intent has consequence=0.70 which exceeds default 0.30
    let router = env.router_with(
        RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 0.30 },
        0.0,
        MockBackend::local("local review", 0.99),
        MockBackend::api("api review"),
    );

    // "review" intent triggers consequence > 0.30
    let req = user_req("please review this Rust code for best practices and security issues");
    let r   = router.route(&req).await.unwrap();
    // High consequence → API
    assert_eq!(r.decision, RouteDecision::Api);
}

#[tokio::test]
async fn second_call_after_local_hit_serves_from_cache() {
    let env    = Env::new().await;
    let router = env.router_force_local(0.80);

    let req = user_req("how does Rust handle memory without a garbage collector");

    // First call — local model serves and caches
    let r1 = router.route(&req).await.unwrap();
    assert_eq!(r1.decision, RouteDecision::LocalModel);

    // Second call — must come from cache now
    let r2 = router.route(&req).await.unwrap();
    assert_eq!(r2.decision, RouteDecision::ExactCache);
}

#[tokio::test]
async fn streaming_request_flag_does_not_break_routing() {
    let env    = Env::new().await;
    let router = env.router_force_api();

    let mut req = user_req("explain async/await in Rust");
    req.stream  = Some(true);

    // Router handles stream by setting it to false internally for cache/local path
    let r = router.route(&req).await.unwrap();
    // Should still route correctly (stream=true goes through the same cascade in non-stream mode)
    assert!(matches!(r.decision, RouteDecision::Api | RouteDecision::ExactCache | RouteDecision::LocalModel));
}

// ── Domain classification tests ───────────────────────────────────────────────

#[test]
fn domain_classify_rust() {
    let shape = claude_cache::domain::classify("how do I use async fn in Rust with tokio");
    assert_eq!(shape.domain, "rust");
}

#[test]
fn domain_classify_python() {
    let shape = claude_cache::domain::classify("how do I use pandas to read a csv file in python");
    assert_eq!(shape.domain, "python");
}

#[test]
fn domain_classify_sql() {
    let shape = claude_cache::domain::classify("write a SELECT query with a LEFT JOIN in SQL");
    assert_eq!(shape.domain, "sql");
}

#[test]
fn domain_classify_shell() {
    let shape = claude_cache::domain::classify("write a bash script to grep logs and count errors");
    assert_eq!(shape.domain, "shell");
}

#[test]
fn intent_classify_fix() {
    let shape = claude_cache::domain::classify("there's a bug in my Rust code where the borrow checker fails");
    assert_eq!(shape.intent, "fix");
}

#[test]
fn intent_classify_explain() {
    let shape = claude_cache::domain::classify("explain what is a monad in Haskell");
    assert_eq!(shape.intent, "explain");
}

#[test]
fn intent_classify_generate() {
    let shape = claude_cache::domain::classify("write a function to parse JSON in Rust using serde");
    assert_eq!(shape.intent, "generate");
}

#[test]
fn complexity_high_for_architecture_prompts() {
    let shape = claude_cache::domain::classify(
        "design a distributed consensus algorithm architecture for a Rust microservice with async concurrent transactions"
    );
    assert!(shape.complexity > 0.5, "complexity={} should be > 0.5", shape.complexity);
}

#[test]
fn complexity_low_for_hello_world() {
    let shape = claude_cache::domain::classify("write a simple hello world example in Python");
    assert!(shape.complexity < 0.5, "complexity={} should be < 0.5", shape.complexity);
}

// ── Cache key / content-addressing tests ─────────────────────────────────────

#[test]
fn cache_key_is_deterministic() {
    let k1 = CacheStore::content_key("what is rust");
    let k2 = CacheStore::content_key("what is rust");
    assert_eq!(k1, k2);
}

#[test]
fn cache_key_normalizes_whitespace() {
    let k1 = CacheStore::content_key("what  is   rust");
    let k2 = CacheStore::content_key("what is rust");
    assert_eq!(k1, k2, "extra whitespace should be normalized");
}

#[test]
fn cache_key_normalizes_case() {
    let k1 = CacheStore::content_key("What Is Rust");
    let k2 = CacheStore::content_key("what is rust");
    assert_eq!(k1, k2, "case should be normalized");
}

#[test]
fn cache_key_differs_for_different_prompts() {
    let k1 = CacheStore::content_key("what is rust");
    let k2 = CacheStore::content_key("what is python");
    assert_ne!(k1, k2);
}
