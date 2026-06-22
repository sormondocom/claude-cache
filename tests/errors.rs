//! Integration tests for the ProxyError → HTTP error code dispatch.
//!
//! Each test uses a real Axum server with an error-injecting mock API backend,
//! exercising the full path: server → router → backend → route_error → HTTP response.
//! Verifies status codes, Retry-After headers, and structured JSON error bodies.

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use arc_swap::ArcSwap;
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use tempfile::tempdir;

use claude_cache::{
    auth::CredentialStore,
    backend::{
        AnthropicBackend, BackendResult, ContentBlock,
        MessagesRequest, MessagesResponse, ModelBackend, Usage,
    },
    budget::BudgetLedger,
    cache::CacheStore,
    config::{ApiConfig, AppConfig, BudgetConfig, RoutingConfig},
    embedding::StubEmbedder,
    error::ProxyError,
    federation::FederationClient,
    identity::NodeIdentity,
    learning::{CalibrationMap, Distiller, ThresholdMap},
    router::Router,
    server::{AppState, build_router},
    trust::TrustStore,
};

// ── Error-injecting mock backend ──────────────────────────────────────────────

struct ErrorBackend {
    make_err: Box<dyn Fn() -> anyhow::Error + Send + Sync>,
}

impl ErrorBackend {
    fn returning<F: Fn() -> anyhow::Error + Send + Sync + 'static>(f: F) -> Arc<Self> {
        Arc::new(Self { make_err: Box::new(f) })
    }
    fn at_capacity() -> Arc<Self> {
        Self::returning(|| ProxyError::BackendAtCapacity(
            "all 4 process slot(s) occupied for >30s".into()
        ).into())
    }
    fn timeout() -> Arc<Self> {
        Self::returning(|| ProxyError::BackendTimeout(
            "claude CLI timed out after 300s".into()
        ).into())
    }
    fn no_api_access() -> Arc<Self> {
        Self::returning(|| ProxyError::NoApiAccess(
            "OAuth token lacks direct API access".into()
        ).into())
    }
    fn rate_limited() -> Arc<Self> {
        Self::returning(|| ProxyError::RateLimited(
            "rate_limit_error from Anthropic".into()
        ).into())
    }
    fn unavailable() -> Arc<Self> {
        Self::returning(|| ProxyError::BackendUnavailable(
            "claude not found in PATH".into()
        ).into())
    }
}

#[async_trait]
impl ModelBackend for ErrorBackend {
    async fn complete(&self, _req: &MessagesRequest) -> Result<BackendResult> {
        Err((self.make_err)())
    }
    fn name(&self) -> &'static str { "mock-error" }
}

// ── Success stub (used for local/distiller roles that aren't under test) ──────

struct StubOkBackend;

#[async_trait]
impl ModelBackend for StubOkBackend {
    async fn complete(&self, _req: &MessagesRequest) -> Result<BackendResult> {
        Ok(BackendResult {
            response: MessagesResponse {
                id: "stub".into(), kind: "message".into(), role: "assistant".into(),
                content: vec![ContentBlock {
                    kind: "text".into(), text: Some("stub ok".into()),
                    extra: Default::default(),
                }],
                model: "stub".into(), stop_reason: Some("end_turn".into()),
                usage: Usage { input_tokens: 1, output_tokens: 1 },
            },
            confidence: None, latency_ms: 1,
        })
    }
    fn name(&self) -> &'static str { "stub-ok" }
}

// ── Server builder ────────────────────────────────────────────────────────────

struct ErrorServerParams {
    /// The API backend — always an ErrorBackend or StubOkBackend.
    api:              Arc<dyn ModelBackend>,
    /// Daily budget limit.  0.0 triggers CC-E007 (budget exceeded).
    daily_limit_usd:  f64,
    /// Routing thresholds.  Use (0.0, 0.0, 0.0) to route directly to API;
    /// (1.0, 1.0, 1.0) to let the budget/local gate run first.
    routing:          RoutingConfig,
}

impl ErrorServerParams {
    /// Route every cold request directly to the API backend.
    /// Use this for all typed ProxyError tests except BudgetExceeded.
    fn direct_to_api(api: Arc<dyn ModelBackend>) -> Self {
        ErrorServerParams {
            api,
            daily_limit_usd: 100.0,
            routing: RoutingConfig {
                novelty_threshold:     0.0,
                complexity_threshold:  0.0,
                consequence_threshold: 0.0,
                draft_verify_enabled:  false,
                draft_verify_min_sim:  0.65,
            },
        }
    }

    /// Trigger the budget-exceeded path (daily_limit = 0, local disabled).
    fn budget_exceeded(api: Arc<dyn ModelBackend>) -> Self {
        ErrorServerParams {
            api,
            daily_limit_usd: 0.0,
            routing: RoutingConfig {
                novelty_threshold:     1.0,
                complexity_threshold:  1.0,
                consequence_threshold: 1.0,
                draft_verify_enabled:  false,
                draft_verify_min_sim:  0.65,
            },
        }
    }
}

async fn start_error_server(p: ErrorServerParams) -> (String, tempfile::TempDir) {
    let dir      = tempdir().unwrap();
    let identity = Arc::new(NodeIdentity::generate());

    let cache_path  = dir.path().join("cache.db").to_str().unwrap().to_string();
    let trust_path  = dir.path().join("trust.db").to_str().unwrap().to_string();
    let budget_path = dir.path().join("budget.db").to_str().unwrap().to_string();

    let cache  = Arc::new(CacheStore::open(&cache_path, &identity.fingerprint).await.unwrap());
    let trust  = Arc::new(TrustStore::open(&trust_path, &identity.fingerprint).await.unwrap());
    let budget = Arc::new(BudgetLedger::open(BudgetConfig {
        enabled:           true,
        db_path:           budget_path,
        daily_limit_usd:   p.daily_limit_usd,
        warn_at_pct:       80,
        input_per_1k_usd:  0.003,
        output_per_1k_usd: 0.015,
    }).await.unwrap());

    let mut cfg = AppConfig::default();
    cfg.routing           = p.routing;
    cfg.local.enabled     = false; // no local-model fallback in error tests
    cfg.embedding.enabled = false;
    let cfg_arc = Arc::new(ArcSwap::from_pointee(cfg));

    let stub_local = Arc::new(StubOkBackend);
    let router = Router::new(
        cfg_arc.clone(),
        cache.clone(),
        budget.clone(),
        Arc::new(StubEmbedder::new(64)),
        stub_local.clone(),
        p.api,
        Arc::new(ArcSwap::from_pointee(ThresholdMap::new())),
        Arc::new(ArcSwap::from_pointee(CalibrationMap::new())),
    );

    let federation = Arc::new(FederationClient::new(false, identity.clone(), trust.clone(), 500));
    let dummy_api_cfg = ApiConfig {
        model:                          "test-model".to_string(),
        base_url:                       "http://127.0.0.1:1".to_string(),
        backend:                        "anthropic".to_string(),
        enabled:                        true,
        max_retries:                    0,
        retry_delay_ms:                 500,
        request_timeout_secs:           30,
        claude_code_max_concurrency:    4,
        claude_code_queue_timeout_secs: 30,
    };
    let anthropic = Arc::new(AnthropicBackend::new(
        &dummy_api_cfg, CredentialStore::from_key("sk-ant-test-key"),
    ));
    let distiller = Arc::new(Distiller::new(cache.clone(), cfg_arc.clone(), stub_local));
    let node_id   = identity.fingerprint.clone();

    let state = Arc::new(AppState {
        router, cache, budget, federation, trust, identity, anthropic,
        cfg:                cfg_arc,
        config_path:        "test.toml".to_string(),
        node_id,
        is_cnc:             false,
        auto_promote_peers: false,
        api_base_url:       "http://127.0.0.1:1".to_string(),
        api_creds:          CredentialStore::from_key("sk-ant-test-key"),
        portal_token:       None,
        rate_limit_rpm:     0,
        credits_exhausted:  AtomicBool::new(false),
        manual_bypass:      AtomicBool::new(false),
        distiller,
        graph_cache:        tokio::sync::Mutex::new(None),
        http_client:        reqwest::Client::new(),
    });

    let app      = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr     = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), dir)
}

fn user_msg(text: &str) -> Value {
    json!({ "model": "claude-sonnet-4-6", "messages": [{ "role": "user", "content": text }], "max_tokens": 1024 })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Pool full: 503 Service Unavailable + Retry-After: 5 + CC-E001 body.
#[tokio::test]
async fn backend_at_capacity_returns_503_with_retry_after() {
    let (base, _dir) = start_error_server(
        ErrorServerParams::direct_to_api(ErrorBackend::at_capacity())
    ).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("pool overload test"))
        .send().await.unwrap();

    assert_eq!(resp.status(), 503, "pool overload must return 503 Service Unavailable");
    assert_eq!(
        resp.headers().get("retry-after").and_then(|v| v.to_str().ok()),
        Some("5"),
        "CC-E001 must include Retry-After: 5"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "error",                        "top-level type must be 'error'");
    assert_eq!(body["error"]["type"], "backend_at_capacity", "error.type must match variant name");
    assert_eq!(body["error"]["code"], "CC-E001",             "error.code must be stable CC-E001");
    assert!(
        body["error"]["message"].as_str().unwrap().contains("CC-E001"),
        "error.message must embed the error code for log correlation"
    );
}

/// Request timeout: 504 Gateway Timeout + Retry-After: 10 + CC-E002 body.
#[tokio::test]
async fn backend_timeout_returns_504_with_retry_after() {
    let (base, _dir) = start_error_server(
        ErrorServerParams::direct_to_api(ErrorBackend::timeout())
    ).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("timeout test"))
        .send().await.unwrap();

    assert_eq!(resp.status(), 504, "timeout must return 504 Gateway Timeout");
    assert_eq!(
        resp.headers().get("retry-after").and_then(|v| v.to_str().ok()),
        Some("10"),
        "CC-E002 must include Retry-After: 10"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "backend_timeout");
    assert_eq!(body["error"]["code"], "CC-E002");
    assert!(body["error"]["message"].as_str().unwrap().contains("CC-E002"));
}

/// OAuth token without API access: 401 Unauthorized, NO Retry-After (permanent).
#[tokio::test]
async fn no_api_access_returns_401_without_retry_after() {
    let (base, _dir) = start_error_server(
        ErrorServerParams::direct_to_api(ErrorBackend::no_api_access())
    ).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("no-access test"))
        .send().await.unwrap();

    assert_eq!(resp.status(), 401, "no_api_access must return 401 Unauthorized");
    assert!(
        resp.headers().get("retry-after").is_none(),
        "CC-E003 must NOT include Retry-After — this is a permanent auth error, not a transient one"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "no_api_access");
    assert_eq!(body["error"]["code"], "CC-E003");
}

/// Anthropic rate limit: 429 Too Many Requests + Retry-After: 60 + CC-E004 body.
#[tokio::test]
async fn rate_limited_returns_429_with_retry_after() {
    let (base, _dir) = start_error_server(
        ErrorServerParams::direct_to_api(ErrorBackend::rate_limited())
    ).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("rate limit test"))
        .send().await.unwrap();

    assert_eq!(resp.status(), 429, "rate_limited must return 429 Too Many Requests");
    assert_eq!(
        resp.headers().get("retry-after").and_then(|v| v.to_str().ok()),
        Some("60"),
        "CC-E004 must include Retry-After: 60"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "rate_limited");
    assert_eq!(body["error"]["code"], "CC-E004");
}

/// `claude` CLI missing from PATH: 503 + NO Retry-After (fix PATH, not a transient issue).
#[tokio::test]
async fn backend_unavailable_returns_503_without_retry_after() {
    let (base, _dir) = start_error_server(
        ErrorServerParams::direct_to_api(ErrorBackend::unavailable())
    ).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("unavailable test"))
        .send().await.unwrap();

    assert_eq!(resp.status(), 503, "backend_unavailable must return 503 Service Unavailable");
    assert!(
        resp.headers().get("retry-after").is_none(),
        "CC-E006 must NOT include Retry-After — fix the PATH, retrying won't help"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "backend_unavailable");
    assert_eq!(body["error"]["code"], "CC-E006");
}

/// Daily budget exhausted: 429 with CC-E007 in body + Retry-After: 300.
/// Budget exceeded is produced by the router's budget gate (not a backend error),
/// so this uses the budget_exceeded server params (thresholds=1.0, limit=0.0).
#[tokio::test]
async fn budget_exceeded_returns_429_with_cc_e007() {
    // StubOkBackend is the API backend here; it's never reached because the
    // budget gate blocks the request before Step 8.
    let (base, _dir) = start_error_server(
        ErrorServerParams::budget_exceeded(Arc::new(StubOkBackend))
    ).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("budget exceeded body test"))
        .send().await.unwrap();

    assert_eq!(resp.status(), 429, "budget exceeded must return 429");
    assert_eq!(
        resp.headers().get("retry-after").and_then(|v| v.to_str().ok()),
        Some("300"),
        "CC-E007 must include Retry-After: 300"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"],           "error",          "body.type must be 'error'");
    assert_eq!(body["error"]["type"],  "budget_exceeded", "error.type must be 'budget_exceeded'");
    assert_eq!(body["error"]["code"],  "CC-E007",         "error.code must be stable CC-E007");
    assert!(
        body["error"]["message"].as_str().is_some(),
        "error.message must be present"
    );
}

/// All structured error responses must have the required three fields.
/// This meta-test guards against regressions in route_error's JSON body builder.
#[tokio::test]
async fn error_body_always_has_type_error_and_code_fields() {
    // Use BackendAtCapacity as the representative case.
    let (base, _dir) = start_error_server(
        ErrorServerParams::direct_to_api(ErrorBackend::at_capacity())
    ).await;

    let body: Value = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&user_msg("structure check"))
        .send().await.unwrap()
        .json().await.unwrap();

    // Top-level envelope
    assert!(body.is_object(),                      "response body must be a JSON object");
    assert!(body["type"].is_string(),              "body.type must be present");
    // Error sub-object
    assert!(body["error"].is_object(),             "body.error must be a JSON object");
    assert!(body["error"]["type"].is_string(),     "error.type must be a string");
    assert!(body["error"]["code"].is_string(),     "error.code must be a string");
    assert!(body["error"]["message"].is_string(),  "error.message must be a string");

    // Code must match the CC-Exxx pattern
    let code = body["error"]["code"].as_str().unwrap();
    assert!(code.starts_with("CC-E"), "error.code must start with 'CC-E', got '{code}'");
}
