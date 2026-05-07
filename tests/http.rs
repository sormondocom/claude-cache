//! HTTP integration tests — real Axum server bound to a random port.
//! Uses mock router backends; no real LLM or Anthropic API calls are made.

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::tempdir;

use claude_cache::{
    auth::Credentials,
    backend::{
        AnthropicBackend, BackendResult, ContentBlock, Message, MessageContent,
        MessagesRequest, MessagesResponse, ModelBackend, Usage,
    },
    budget::BudgetLedger,
    cache::CacheStore,
    config::{ApiConfig, AppConfig, BudgetConfig, RoutingConfig},
    embedding::StubEmbedder,
    federation::FederationClient,
    identity::NodeIdentity,
    router::Router,
    server::{AppState, build_router},
    trust::TrustStore,
};

// ── Mock backends ─────────────────────────────────────────────────────────────

struct MockBackend {
    name_str:   &'static str,
    confidence: Option<f64>,
    text:       &'static str,
}

#[async_trait]
impl ModelBackend for MockBackend {
    async fn complete(&self, _req: &MessagesRequest) -> Result<BackendResult> {
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

// ── Test server builder ───────────────────────────────────────────────────────

struct ServerParams {
    daily_limit_usd:   f64,
    routing:           RoutingConfig,
    local_confidence:  f64,
    confidence_floor:  f64,
    portal_token:      Option<String>,
}

impl ServerParams {
    fn force_api(daily_limit: f64) -> Self {
        ServerParams {
            daily_limit_usd:  daily_limit,
            routing:          RoutingConfig { novelty_threshold: 0.0, complexity_threshold: 0.0, consequence_threshold: 0.0 },
            local_confidence: 0.9,
            confidence_floor: 0.5,
            portal_token:     None,
        }
    }

    fn force_local(daily_limit: f64) -> Self {
        ServerParams {
            daily_limit_usd:  daily_limit,
            routing:          RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0 },
            local_confidence: 0.9,
            confidence_floor: 0.5,
            portal_token:     None,
        }
    }
}

/// Start a test server and return its base URL.
/// The returned `TempDir` must be kept alive for the lifetime of the test.
async fn start_server(p: ServerParams) -> (String, tempfile::TempDir) {
    let dir      = tempdir().unwrap();
    let identity = Arc::new(NodeIdentity::generate());

    let cache_path  = dir.path().join("cache.db").to_str().unwrap().to_string();
    let trust_path  = dir.path().join("trust.db").to_str().unwrap().to_string();
    let budget_path = dir.path().join("budget.db").to_str().unwrap().to_string();

    let cache  = Arc::new(CacheStore::open(&cache_path,  &identity.fingerprint).await.unwrap());
    let trust  = Arc::new(TrustStore::open(&trust_path,  &identity.fingerprint).await.unwrap());
    let budget = Arc::new(BudgetLedger::open(BudgetConfig {
        db_path:           budget_path,
        daily_limit_usd:   p.daily_limit_usd,
        warn_at_pct:       80,
        input_per_1k_usd:  0.003,
        output_per_1k_usd: 0.015,
    }).await.unwrap());

    let mut cfg = AppConfig::default();
    cfg.routing                = p.routing;
    cfg.local.enabled          = true;
    cfg.local.confidence_floor = p.confidence_floor;
    cfg.embedding.enabled      = false;
    let cfg = Arc::new(cfg);

    let router = Router::new(
        cfg.clone(),
        cache.clone(),
        budget.clone(),
        Arc::new(StubEmbedder::new(64)),
        Arc::new(MockBackend { name_str: "mock-local", confidence: Some(p.local_confidence), text: "local answer" }),
        Arc::new(MockBackend { name_str: "mock-api",   confidence: None,                     text: "api answer"   }),
    );

    let federation = Arc::new(FederationClient::new(
        vec![], false, identity.clone(), trust.clone(), 500,
    ));

    // AnthropicBackend is only used for streaming passthrough; point it at
    // an unreachable address so non-streaming tests never touch it.
    let dummy_api_cfg = ApiConfig {
        model:    "test-model".to_string(),
        base_url: "http://127.0.0.1:1".to_string(),
    };
    let dummy_creds = Credentials { api_key: "sk-ant-test-key".to_string() };
    let anthropic = Arc::new(AnthropicBackend::new(&dummy_api_cfg, dummy_creds.clone()));

    let node_id = identity.fingerprint.clone();
    let state = Arc::new(AppState {
        router,
        cache,
        budget,
        federation,
        trust,
        identity,
        anthropic,
        node_id,
        is_cnc:             false,
        auto_promote_peers: false,
        api_base_url:       "http://127.0.0.1:1".to_string(),
        api_creds:          dummy_creds,
        portal_token:       p.portal_token,
    });

    let app      = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr     = listener.local_addr().unwrap();

    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    (format!("http://{addr}"), dir)
}

fn user_msg(text: &str) -> Value {
    json!({
        "model":      "claude-sonnet-4-6",
        "messages":   [{ "role": "user", "content": text }],
        "max_tokens": 1024
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_200_with_node_id() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let resp = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert!(body["node_id"].as_str().unwrap().len() > 10, "node_id should be a fingerprint");
    assert_eq!(body["federation_peers"], 0);
}

#[tokio::test]
async fn stats_returns_expected_structure() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let body: Value = reqwest::get(format!("{base}/stats"))
        .await.unwrap()
        .json().await.unwrap();
    assert!(body["cache"].is_object(),  "stats.cache must be an object");
    assert!(body["budget"].is_object(), "stats.budget must be an object");
    assert!(body["node_id"].is_string(), "stats.node_id must be a string");
    assert_eq!(body["cache"]["entries"], 0);
}

#[tokio::test]
async fn messages_cache_miss_routes_to_api() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("what is a Rust lifetime?"))
        .send().await.unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers().get("x-router-source").and_then(|v| v.to_str().ok()),
        Some("api"),
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["content"][0]["text"], "api answer");
}

#[tokio::test]
async fn messages_second_call_hits_cache() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let client = reqwest::Client::new();
    let payload = user_msg("explain Rust ownership in detail");

    // First call — api
    let r1 = client.post(format!("{base}/v1/messages")).json(&payload).send().await.unwrap();
    assert_eq!(r1.status(), 200);
    assert_eq!(r1.headers()["x-router-source"], "api");
    let _ = r1.bytes().await; // consume body

    // Second call — exact cache hit
    let r2 = client.post(format!("{base}/v1/messages")).json(&payload).send().await.unwrap();
    assert_eq!(r2.status(), 200);
    assert_eq!(r2.headers()["x-router-source"], "exact_cache");
    let body: Value = r2.json().await.unwrap();
    assert_eq!(body["content"][0]["text"], "api answer");
}

#[tokio::test]
async fn messages_local_model_served_when_forced() {
    let (base, _dir) = start_server(ServerParams::force_local(10.0)).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("explain Rust closures briefly"))
        .send().await.unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["x-router-source"], "local");
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["content"][0]["text"], "local answer");
}

#[tokio::test]
async fn trust_list_returns_empty_nodes_array() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let body: Value = reqwest::get(format!("{base}/v1/trust"))
        .await.unwrap()
        .json().await.unwrap();
    assert!(body["nodes"].is_array(), "/v1/trust must return a nodes array");
}

#[tokio::test]
async fn portal_api_trust_returns_nodes() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let body: Value = reqwest::get(format!("{base}/api/trust"))
        .await.unwrap()
        .json().await.unwrap();
    assert!(body["nodes"].is_array(), "/api/trust must return a nodes array");
}

#[tokio::test]
async fn portal_overview_has_expected_fields() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let body: Value = reqwest::get(format!("{base}/api/overview"))
        .await.unwrap()
        .json().await.unwrap();
    assert!(body["node_id"].is_string());
    assert!(body["cache"].is_object());
    assert!(body["budget"].is_object());
    assert!(body["federation"].is_object());
    assert_eq!(body["budget"]["status"], "ok");
}

#[tokio::test]
async fn budget_exceeded_blocks_api_call() {
    // daily_limit=0 → always exceeded; local mock returns 0.3 confidence < floor 0.5
    let (base, _dir) = start_server(ServerParams {
        daily_limit_usd:  0.0,
        routing:          RoutingConfig { novelty_threshold: 1.0, complexity_threshold: 1.0, consequence_threshold: 1.0 },
        local_confidence: 0.3,  // below floor
        confidence_floor: 0.5,
        portal_token:     None,
    }).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("write me a web server in Rust"))
        .send().await.unwrap();

    // Must NOT return "api answer" — the API call must be blocked
    assert_eq!(resp.status(), 500, "budget exceeded with no local fallback must return 500");
}

#[tokio::test]
async fn dashboard_html_is_served() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let resp = reqwest::get(format!("{base}/")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("claude-cache"), "dashboard must contain page title");
    assert!(text.contains("Trust"), "dashboard must contain trust section");
}

#[tokio::test]
async fn peer_health_endpoint_returns_empty_when_no_checks_recorded() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let body: Value = reqwest::get(format!("{base}/api/peers/health"))
        .await.unwrap()
        .json().await.unwrap();
    assert!(body["peers"].is_array(), "/api/peers/health must return a peers array");
    assert_eq!(body["peers"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn peer_health_record_and_retrieve() {
    let dir      = tempfile::tempdir().unwrap();
    let trust_path = dir.path().join("trust.db").to_str().unwrap().to_string();
    let identity = claude_cache::identity::NodeIdentity::generate();
    let trust = std::sync::Arc::new(
        claude_cache::trust::TrustStore::open(&trust_path, &identity.fingerprint).await.unwrap()
    );

    // Record a successful check at 42ms
    trust.record_health_check("peer-abc", "http://peer:3000", true, Some(42), 3).await.unwrap();
    let health = trust.list_peer_health().await.unwrap();
    assert_eq!(health.len(), 1);
    assert_eq!(health[0].node_id, "peer-abc");
    assert!(health[0].is_reachable);
    assert_eq!(health[0].latency_ms, Some(42));
    assert_eq!(health[0].consecutive_fail, 0);
    assert_eq!(health[0].check_count, 1);

    // Record two consecutive failures — below threshold of 3, still reachable
    trust.record_health_check("peer-abc", "http://peer:3000", false, None, 3).await.unwrap();
    trust.record_health_check("peer-abc", "http://peer:3000", false, None, 3).await.unwrap();
    let health = trust.list_peer_health().await.unwrap();
    assert!(health[0].is_reachable, "2 failures < threshold of 3 → still reachable");
    assert_eq!(health[0].consecutive_fail, 2);

    // Third failure hits threshold → unreachable
    trust.record_health_check("peer-abc", "http://peer:3000", false, None, 3).await.unwrap();
    let health = trust.list_peer_health().await.unwrap();
    assert!(!health[0].is_reachable, "3 failures == threshold → unreachable");

    // Recovery: one success restores reachability
    trust.record_health_check("peer-abc", "http://peer:3000", true, Some(55), 3).await.unwrap();
    let health = trust.list_peer_health().await.unwrap();
    assert!(health[0].is_reachable, "successful check → reachable again");
    assert_eq!(health[0].consecutive_fail, 0);
}

#[tokio::test]
async fn federation_semantic_endpoint_returns_empty_when_no_embeddings() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let client = reqwest::Client::new();

    // Embedding dims match the stub embedder (64-d).
    let embedding: Vec<f32> = vec![0.0f32; 64];
    let body = json!({
        "domain":        "rust",
        "embedding":     embedding,
        "sim_threshold": 0.85,
        "limit":         3,
    });

    let resp = client
        .post(format!("{base}/v1/federation/semantic"))
        .json(&body)
        .send().await.unwrap();

    assert_eq!(resp.status(), 200);
    let arr: Value = resp.json().await.unwrap();
    assert!(arr.is_array(), "/v1/federation/semantic must return a JSON array");
    // No embeddings stored yet — must be empty.
    assert_eq!(arr.as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn routing_log_returns_summary_and_recent() {
    let (base, _dir) = start_server(ServerParams::force_api(10.0)).await;
    let client = reqwest::Client::new();

    // Make a couple of requests so the log is populated.
    client.post(format!("{base}/v1/messages")).json(&user_msg("hello there")).send().await.unwrap();
    client.post(format!("{base}/v1/messages")).json(&user_msg("hello there")).send().await.unwrap();

    let body: Value = reqwest::get(format!("{base}/api/routing"))
        .await.unwrap()
        .json().await.unwrap();
    assert!(body["summary"].is_object(), "/api/routing must have a summary");
    assert!(body["recent"].is_array(),   "/api/routing must have a recent array");
    assert!(body["summary"]["total_requests"].as_i64().unwrap_or(0) >= 2);
}

#[tokio::test]
async fn portal_auth_blocks_without_token() {
    let (base, _dir) = start_server(ServerParams {
        portal_token: Some("secret123".to_string()),
        ..ServerParams::force_api(10.0)
    }).await;

    // Protected endpoint without token → 401.
    let resp = reqwest::get(format!("{base}/stats")).await.unwrap();
    assert_eq!(resp.status(), 401);

    // Dashboard also blocked.
    let resp = reqwest::get(format!("{base}/")).await.unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn portal_auth_allows_with_correct_token() {
    let (base, _dir) = start_server(ServerParams {
        portal_token: Some("secret123".to_string()),
        ..ServerParams::force_api(10.0)
    }).await;

    let client = reqwest::Client::new();

    // Correct token → 200.
    let resp = client
        .get(format!("{base}/stats"))
        .bearer_auth("secret123")
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn portal_auth_does_not_block_messages_or_health() {
    let (base, _dir) = start_server(ServerParams {
        portal_token: Some("secret123".to_string()),
        ..ServerParams::force_api(10.0)
    }).await;

    let client = reqwest::Client::new();

    // /health is always public.
    let resp = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);

    // /v1/messages is always public (clients auth via API key).
    let resp = client
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("auth bypass check"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
}
