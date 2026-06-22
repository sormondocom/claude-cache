//! Integration tests for the routing cascade.
//!
//! Each test uses real async SQLite stores (tempfile) and mock ModelBackend
//! implementations so routing decisions are deterministic without network calls.

use std::sync::Arc;
use anyhow::Result;
use arc_swap::ArcSwap;
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
    learning::{CalibrationMap, ThresholdMap},
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
    async fn complete(&self, _req: &MessagesRequest) -> Result<BackendResult> {
        Ok(BackendResult {
            response: MessagesResponse {
                id:          "test-id".into(),
                kind:        "message".into(),
                role:        "assistant".into(),
                content:     vec![ContentBlock { kind: "text".into(), text: Some(self.text.into()), extra: Default::default() }],
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

/// A backend that always returns an error.  Used to test escalation paths.
struct FailingBackend {
    err_msg: &'static str,
}

#[async_trait]
impl ModelBackend for FailingBackend {
    async fn complete(&self, _req: &MessagesRequest) -> Result<BackendResult> {
        Err(anyhow::anyhow!("{}", self.err_msg))
    }
    fn name(&self) -> &'static str { "mock-failing" }
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
            enabled:           true,
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
        cfg.routing = RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0, draft_verify_enabled: false, draft_verify_min_sim: 0.65 };
        cfg.local.confidence_floor = local_confidence;
        cfg.local.enabled = true;
        cfg.embedding.enabled = false; // stub only in tests

        Router::new(
            Arc::new(ArcSwap::from_pointee(cfg)),
            self.cache.clone(),
            self.budget.clone(),
            Arc::new(StubEmbedder::new(64)),
            Arc::new(MockBackend::local("local answer", 0.90)),
            Arc::new(MockBackend::api("api answer")),
            Arc::new(ArcSwap::from_pointee(ThresholdMap::new())),
            Arc::new(ArcSwap::from_pointee(CalibrationMap::new())),
        )
    }

    /// Build a Router that always routes to API (thresholds all 0.0).
    fn router_force_api(&self) -> Router {
        let mut cfg = (*self.cfg).clone();
        cfg.routing = RoutingConfig { novelty_threshold: 0.0, complexity_threshold: 0.0, consequence_threshold: 0.0, draft_verify_enabled: false, draft_verify_min_sim: 0.65 };
        cfg.local.enabled = true;
        cfg.embedding.enabled = false;

        Router::new(
            Arc::new(ArcSwap::from_pointee(cfg)),
            self.cache.clone(),
            self.budget.clone(),
            Arc::new(StubEmbedder::new(64)),
            Arc::new(MockBackend::local("local answer", 0.90)),
            Arc::new(MockBackend::api("api answer")),
            Arc::new(ArcSwap::from_pointee(ThresholdMap::new())),
            Arc::new(ArcSwap::from_pointee(CalibrationMap::new())),
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
            Arc::new(ArcSwap::from_pointee(cfg)),
            self.cache.clone(),
            self.budget.clone(),
            Arc::new(StubEmbedder::new(64)),
            Arc::new(local),
            Arc::new(api),
            Arc::new(ArcSwap::from_pointee(ThresholdMap::new())),
            Arc::new(ArcSwap::from_pointee(CalibrationMap::new())),
        )
    }
}

fn user_req(text: &str) -> MessagesRequest {
    MessagesRequest {
        model:          "claude-sonnet-4-6".into(),
        messages:       vec![Message { role: "user".into(), content: MessageContent::Text(text.into()) }],
        max_tokens:     1024,
        system:         None,
        stream:         None,
        tools:          None,
        extra:          Default::default(),
        anthropic_beta: None,
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
        RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0, draft_verify_enabled: false, draft_verify_min_sim: 0.65 },
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
        content:     vec![ContentBlock { kind: "text".into(), text: Some("ownership means one owner".into()), extra: Default::default() }],
        model:       "anthropic".into(),
        stop_reason: Some("end_turn".into()),
        usage:       Usage { input_tokens: 5, output_tokens: 10 },
    }).unwrap();

    env.cache.store(&shape, prompt, None, &stored_resp, "anthropic", None, Some(3600), false, false).await.unwrap();

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
    cfg.routing = RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0, draft_verify_enabled: false, draft_verify_min_sim: 0.65 };
    cfg.local.confidence_floor = 0.0; // accept any local confidence
    cfg.local.enabled = true;
    cfg.embedding.enabled = false;
    cfg.budget.daily_limit_usd = 0.0; // already exceeded

    let router = Router::new(
        Arc::new(ArcSwap::from_pointee(cfg)),
        env.cache.clone(),
        env.budget.clone(),
        Arc::new(StubEmbedder::new(64)),
        Arc::new(MockBackend::local("local fallback", 0.50)),
        Arc::new(MockBackend::api("api answer")),
        Arc::new(ArcSwap::from_pointee(ThresholdMap::new())),
        Arc::new(ArcSwap::from_pointee(CalibrationMap::new())),
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
        RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 0.30, draft_verify_enabled: false, draft_verify_min_sim: 0.65 },
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

// ── Error-path cascade tests ──────────────────────────────────────────────────

/// When the local model returns an error (not low confidence, but a crash/IO failure),
/// the router must escalate to the API backend and serve its response.
#[tokio::test]
async fn local_model_error_escalates_to_api() {
    let env = Env::new().await;
    let router = env.router_with(
        RoutingConfig {
            novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0,
            draft_verify_enabled: false, draft_verify_min_sim: 0.65,
        },
        0.0, // confidence floor
        FailingBackend { err_msg: "local model crashed" },
        MockBackend::api("api fallback answer"),
    );

    let req = user_req("explain Rust borrow rules");
    let r   = router.route(&req).await.unwrap();

    assert_eq!(r.decision, RouteDecision::Api,
        "local model error must escalate to API");
    assert_eq!(r.response.text_content(), "api fallback answer");
}

/// When both the local model and API backend fail, the error propagates
/// out of route() as an anyhow::Error (callers must handle it).
#[tokio::test]
async fn both_backends_fail_propagates_error() {
    let env = Env::new().await;
    let router = env.router_with(
        RoutingConfig {
            novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0,
            draft_verify_enabled: false, draft_verify_min_sim: 0.65,
        },
        0.0,
        FailingBackend { err_msg: "local crashed" },
        FailingBackend { err_msg: "api crashed" },
    );

    let req    = user_req("what is a mutex");
    let result = router.route(&req).await;

    assert!(result.is_err(), "both backends failing must propagate an error");
    if let Err(e) = result {
        assert!(!e.to_string().is_empty(), "error message must not be empty");
    }
}

/// When the local model returns Ok but with confidence below the floor, AND
/// the API also fails, the router surfaces the API error to the caller.
#[tokio::test]
async fn low_confidence_local_then_api_fail_propagates_error() {
    let env = Env::new().await;
    let router = env.router_with(
        RoutingConfig {
            novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0,
            draft_verify_enabled: false, draft_verify_min_sim: 0.65,
        },
        0.90, // floor: local returns 0.5 < 0.90 → escalates
        MockBackend::local("weak answer", 0.50),
        FailingBackend { err_msg: "api also down" },
    );

    let req    = user_req("write an async runtime in Rust from scratch");
    let result = router.route(&req).await;

    assert!(result.is_err(), "low-confidence local + API failure must propagate an error");
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

// TypeScript is classified as typescript, not javascript, even when generic JS
// keywords like "const" or "let" are present.
#[test]
fn domain_classify_typescript() {
    let shape = claude_cache::domain::classify(
        "create an interface with optional fields in TypeScript"
    );
    assert_eq!(shape.domain, "typescript");
}

#[test]
fn domain_classify_typescript_beats_javascript() {
    let shape = claude_cache::domain::classify(
        "const greeting: string = 'hello' — how do I annotate types in TypeScript"
    );
    assert_eq!(shape.domain, "typescript",
        "': string' and 'typescript' should beat generic JS keywords like 'const'");
}

// Multi-domain prompts: primary language should win over secondary.
#[test]
fn domain_classify_multi_domain_primary_wins() {
    let shape = claude_cache::domain::classify(
        "write a Python script using pandas to query a PostgreSQL database"
    );
    assert_eq!(shape.domain, "python",
        "Python-specific signals (pandas, def, self) should dominate SQL keywords");
}

// ── Cache key / content-addressing tests ─────────────────────────────────────

#[test]
fn cache_key_is_deterministic() {
    let k1 = CacheStore::content_key("what is rust", None);
    let k2 = CacheStore::content_key("what is rust", None);
    assert_eq!(k1, k2);
}

#[test]
fn cache_key_normalizes_whitespace() {
    let k1 = CacheStore::content_key("what  is   rust", None);
    let k2 = CacheStore::content_key("what is rust", None);
    assert_eq!(k1, k2, "extra whitespace should be normalized");
}

#[test]
fn cache_key_normalizes_case() {
    let k1 = CacheStore::content_key("What Is Rust", None);
    let k2 = CacheStore::content_key("what is rust", None);
    assert_eq!(k1, k2, "case should be normalized");
}

#[test]
fn cache_key_differs_for_different_prompts() {
    let k1 = CacheStore::content_key("what is rust", None);
    let k2 = CacheStore::content_key("what is python", None);
    assert_ne!(k1, k2);
}

#[test]
fn cache_key_differs_with_different_system_prompts() {
    let k_no_sys  = CacheStore::content_key("explain closures", None);
    let k_sys_a   = CacheStore::content_key("explain closures", Some("You are a strict code reviewer."));
    let k_sys_b   = CacheStore::content_key("explain closures", Some("You are a friendly tutor."));
    assert_ne!(k_no_sys, k_sys_a, "system vs no-system must differ");
    assert_ne!(k_sys_a,  k_sys_b, "different system prompts must differ");
}

#[test]
fn cache_key_same_system_prompt_matches() {
    let k1 = CacheStore::content_key("explain closures", Some("You are a strict code reviewer."));
    let k2 = CacheStore::content_key("explain closures", Some("You are a strict code reviewer."));
    assert_eq!(k1, k2, "identical system prompts must produce same key");
}

#[tokio::test]
async fn system_prompt_isolates_cache_entries() {
    let env = Env::new().await;
    let shape = ShapeKey { domain: "rust".into(), intent: "explain".into(), complexity: 0.3 };
    let prompt = "explain closures";

    let resp_a = serde_json::to_string(&MessagesResponse {
        id: "a".into(), kind: "message".into(), role: "assistant".into(),
        content: vec![ContentBlock { kind: "text".into(), text: Some("strict answer".into()), extra: Default::default() }],
        model: "anthropic".into(), stop_reason: Some("end_turn".into()),
        usage: Usage { input_tokens: 5, output_tokens: 10 },
    }).unwrap();
    let resp_b = serde_json::to_string(&MessagesResponse {
        id: "b".into(), kind: "message".into(), role: "assistant".into(),
        content: vec![ContentBlock { kind: "text".into(), text: Some("friendly answer".into()), extra: Default::default() }],
        model: "anthropic".into(), stop_reason: Some("end_turn".into()),
        usage: Usage { input_tokens: 5, output_tokens: 10 },
    }).unwrap();

    env.cache.store(&shape, prompt, Some("strict reviewer"), &resp_a, "anthropic", None, Some(3600), false, false).await.unwrap();
    env.cache.store(&shape, prompt, Some("friendly tutor"),  &resp_b, "anthropic", None, Some(3600), false, false).await.unwrap();

    // Each system prompt should retrieve its own cached response
    let hit_a = env.cache.lookup_exact(prompt, Some("strict reviewer")).await.unwrap().unwrap();
    let hit_b = env.cache.lookup_exact(prompt, Some("friendly tutor")).await.unwrap().unwrap();
    assert!(hit_a.response.contains("strict answer"),   "wrong entry for strict system");
    assert!(hit_b.response.contains("friendly answer"), "wrong entry for friendly system");

    // No system prompt → no hit (different key)
    let hit_none = env.cache.lookup_exact(prompt, None).await.unwrap();
    assert!(hit_none.is_none(), "no-system query must not match system-prompt entry");
}

#[tokio::test]
async fn pinned_entry_survives_eviction() {
    let env = Env::new().await;
    let shape = ShapeKey { domain: "rust".into(), intent: "explain".into(), complexity: 0.3 };

    let make_resp = |id: &str, text: &str| serde_json::to_string(&MessagesResponse {
        id: id.into(), kind: "message".into(), role: "assistant".into(),
        content: vec![ContentBlock { kind: "text".into(), text: Some(text.into()), extra: Default::default() }],
        model: "anthropic".into(), stop_reason: Some("end_turn".into()),
        usage: Usage { input_tokens: 5, output_tokens: 10 },
    }).unwrap();

    // Store a pinned entry with a very short TTL
    let pinned_resp = make_resp("pinned-1", "pinned answer");
    let pinned_id = env.cache.store(&shape, "pinned prompt", None, &pinned_resp, "anthropic", None, Some(1), false, true).await.unwrap();

    // Store some regular (unpinned) entries with expired TTL
    for i in 0..3 {
        let r = make_resp(&format!("exp-{i}"), &format!("expired answer {i}"));
        env.cache.store(&shape, &format!("expired prompt {i}"), None, &r, "anthropic", None, Some(1), false, false).await.unwrap();
    }

    // Evict expired — pinned entry must survive
    env.cache.evict_expired().await.unwrap();

    let still_there = env.cache.lookup_exact("pinned prompt", None).await.unwrap();
    assert!(still_there.is_some(), "pinned entry should survive TTL eviction (id={pinned_id})");
}

#[tokio::test]
async fn search_entries_returns_matching_results() {
    let env = Env::new().await;
    let shape_rust = ShapeKey { domain: "rust".into(), intent: "explain".into(), complexity: 0.3 };
    let shape_py   = ShapeKey { domain: "python".into(), intent: "generate".into(), complexity: 0.4 };

    let dummy_resp = |text: &str| serde_json::to_string(&MessagesResponse {
        id: "x".into(), kind: "message".into(), role: "assistant".into(),
        content: vec![ContentBlock { kind: "text".into(), text: Some(text.into()), extra: Default::default() }],
        model: "anthropic".into(), stop_reason: Some("end_turn".into()),
        usage: Usage { input_tokens: 5, output_tokens: 10 },
    }).unwrap();

    env.cache.store(&shape_rust, "explain lifetimes in Rust",      None, &dummy_resp("a"), "anthropic", None, Some(3600), false, false).await.unwrap();
    env.cache.store(&shape_rust, "explain borrow checker in Rust",  None, &dummy_resp("b"), "anthropic", None, Some(3600), false, false).await.unwrap();
    env.cache.store(&shape_py,   "write a Python pandas script",    None, &dummy_resp("c"), "anthropic", None, Some(3600), false, false).await.unwrap();

    // Search by keyword
    let results = env.cache.search_entries(Some("lifetimes"), None, 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].prompt_preview.contains("lifetimes"));

    // Filter by domain
    let rust_results = env.cache.search_entries(None, Some("rust"), 10).await.unwrap();
    assert_eq!(rust_results.len(), 2);

    // No filter returns all
    let all = env.cache.search_entries(None, None, 10).await.unwrap();
    assert_eq!(all.len(), 3);
}
