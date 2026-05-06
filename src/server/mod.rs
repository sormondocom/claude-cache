use axum::{
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use bytes::Bytes;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn};

use crate::backend::{MessagesRequest, MessagesResponse};
use crate::budget::BudgetLedger;
use crate::cache::CacheStore;
use crate::federation::{entry_to_federated, AnnouncePayload, FederationClient};
use crate::identity::NodeIdentity;
use crate::router::{RouteDecision, Router};
use crate::trust::TrustStore;
use crate::backend::anthropic::AnthropicBackend;

// ── App State ──────────────────────────────────────────────────────────────

pub struct AppState {
    pub router:      Router,
    pub cache:       Arc<CacheStore>,
    pub budget:      Arc<BudgetLedger>,
    pub federation:  Arc<FederationClient>,
    pub trust:       Arc<TrustStore>,
    pub identity:    Arc<NodeIdentity>,
    pub anthropic:   Arc<AnthropicBackend>,
    pub node_id:     String,
    pub api_base_url: String,
    pub api_creds:   crate::auth::Credentials,
}

pub type SharedState = Arc<AppState>;

// ── Routes ─────────────────────────────────────────────────────────────────

pub fn build_router(state: SharedState) -> AxumRouter {
    AxumRouter::new()
        // Primary proxy endpoint
        .route("/v1/messages", post(handle_messages))
        // Health + stats
        .route("/health",      get(handle_health))
        .route("/stats",       get(handle_stats))
        // Budget control
        .route("/api/pricing", post(handle_update_pricing))
        .route("/api/spend",   get(handle_spend))
        // Federation endpoints
        .route("/v1/federation/lookup/:hash",  get(handle_federation_lookup))
        .route("/v1/federation/announce",      post(handle_federation_announce))
        .route("/v1/federation/peers",         get(handle_federation_peers))
        .route("/v1/federation/revocations",   get(handle_get_revocations)
                                               .post(handle_receive_revocation))
        // Trust management
        .route("/v1/trust",                   get(handle_trust_list))
        .route("/v1/trust/:node_id",          post(handle_trust_promote))
        .route("/v1/evict/:node_id",          post(handle_evict))
        // Dashboard
        .route("/",            get(crate::portal::handle_dashboard))
        .route("/api/overview", get(crate::portal::handle_overview))
        .route("/api/cache",   get(crate::portal::handle_cache_entries))
        // Passthrough for any other /v1/* paths
        .fallback(handle_passthrough)
        .with_state(state)
}

// ── /v1/messages ──────────────────────────────────────────────────────────

async fn handle_messages(
    State(state): State<SharedState>,
    _headers:     HeaderMap,
    Json(req):    Json<MessagesRequest>,
) -> Response {
    let is_stream = req.is_streaming();

    if is_stream {
        handle_stream_messages(state, req).await
    } else {
        handle_sync_messages(state, req).await
    }
}

async fn handle_sync_messages(state: SharedState, req: MessagesRequest) -> Response {
    match state.router.route(&req).await {
        Ok(routed) => {
            info!(
                decision = routed.decision.as_str(),
                latency_ms = routed.latency_ms,
                saved_usd = routed.saved_usd,
                "routed"
            );
            let mut resp = Json(routed.response).into_response();
            resp.headers_mut().insert(
                "x-router-source",
                HeaderValue::from_str(routed.decision.as_str()).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
            );
            resp
        }
        Err(e) => {
            warn!("router error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response()
        }
    }
}

async fn handle_stream_messages(state: SharedState, req: MessagesRequest) -> Response {
    // For streaming: check cache/local first (non-stream), then if it's an API
    // call, do a true streaming passthrough.

    // Try the non-stream cache path first
    let non_stream_req = {
        let mut r = req.clone();
        r.stream   = Some(false);
        r
    };

    match state.router.route(&non_stream_req).await {
        Ok(routed) if routed.decision != RouteDecision::Api => {
            // Cache/local hit — synthesize SSE
            info!("stream: synthesizing SSE from {}", routed.decision.as_str());
            return synthesize_sse(routed.response).into_response();
        }
        _ => {}
    }

    // True streaming passthrough to Anthropic
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let anthropic = state.anthropic.clone();
    let req_clone = req.clone();

    tokio::spawn(async move {
        match anthropic.stream_to_channel(&req_clone, tx.clone()).await {
            Ok(acc) => {
                // Background: cache the accumulated response
                // (fire-and-forget, don't block the stream)
                info!(
                    "stream complete: {} output tokens accumulated",
                    acc.output_tokens
                );
            }
            Err(e) => warn!("stream error: {e}"),
        }
    });

    let stream  = ReceiverStream::new(rx);
    let body    = Body::from_stream(stream);
    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-router-source", "api-stream")
        .body(body)
        .unwrap()
}

fn synthesize_sse(resp: MessagesResponse) -> impl IntoResponse {
    use std::fmt::Write;
    let mut sse = String::new();
    let id = &resp.id;

    let _ = writeln!(sse, "event: message_start");
    let _ = writeln!(sse, "data: {}", serde_json::to_string(&json!({
        "type": "message_start",
        "message": { "id": id, "type": "message", "role": "assistant", "model": resp.model, "usage": resp.usage }
    })).unwrap_or_default());
    let _ = writeln!(sse);

    for (i, block) in resp.content.iter().enumerate() {
        let _ = writeln!(sse, "event: content_block_start");
        let _ = writeln!(sse, "data: {}", serde_json::to_string(&json!({
            "type": "content_block_start", "index": i, "content_block": { "type": block.kind, "text": "" }
        })).unwrap_or_default());
        let _ = writeln!(sse);

        if let Some(text) = &block.text {
            let _ = writeln!(sse, "event: content_block_delta");
            let _ = writeln!(sse, "data: {}", serde_json::to_string(&json!({
                "type": "content_block_delta", "index": i, "delta": { "type": "text_delta", "text": text }
            })).unwrap_or_default());
            let _ = writeln!(sse);
        }

        let _ = writeln!(sse, "event: content_block_stop");
        let _ = writeln!(sse, "data: {}", serde_json::json!({ "type": "content_block_stop", "index": i }));
        let _ = writeln!(sse);
    }

    let _ = writeln!(sse, "event: message_delta");
    let _ = writeln!(sse, "data: {}", serde_json::to_string(&json!({
        "type": "message_delta", "delta": { "stop_reason": resp.stop_reason }, "usage": resp.usage
    })).unwrap_or_default());
    let _ = writeln!(sse);
    let _ = writeln!(sse, "event: message_stop");
    let _ = writeln!(sse, "data: {}", serde_json::json!({ "type": "message_stop" }));
    let _ = writeln!(sse);

    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("x-router-source", "cache-sse")
        .body(Body::from(sse))
        .unwrap()
}

// ── Health ─────────────────────────────────────────────────────────────────

async fn handle_health(State(state): State<SharedState>) -> Json<Value> {
    let stats = state.cache.stats().await.unwrap_or(crate::cache::CacheStats {
        total_entries: 0, total_hits: 0, shared_entries: 0, db_size_bytes: 0,
    });
    Json(json!({
        "status": "ok",
        "node_id": state.node_id,
        "cache_entries": stats.total_entries,
        "federation_peers": state.federation.peer_count(),
    }))
}

async fn handle_stats(State(state): State<SharedState>) -> Json<Value> {
    let cache_stats  = state.cache.stats().await.ok();
    let budget_check = state.budget.check().await.ok();
    let summary      = state.budget.daily_summary(7).await.ok();

    Json(json!({
        "node_id": state.node_id,
        "cache":   cache_stats.map(|s| json!({
            "entries": s.total_entries,
            "hits":    s.total_hits,
            "shared":  s.shared_entries,
        })),
        "budget":  budget_check.map(|b| match b {
            crate::budget::BudgetStatus::Ok { spent_usd, limit_usd } =>
                json!({ "status": "ok", "spent_usd": spent_usd, "limit_usd": limit_usd }),
            crate::budget::BudgetStatus::Warning { spent_usd, limit_usd, pct } =>
                json!({ "status": "warning", "spent_usd": spent_usd, "limit_usd": limit_usd, "pct": pct }),
            crate::budget::BudgetStatus::Exceeded { spent_usd, limit_usd } =>
                json!({ "status": "exceeded", "spent_usd": spent_usd, "limit_usd": limit_usd }),
        }),
        "spend_7d": summary,
    }))
}

// ── Budget ─────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct PricingUpdate {
    input_per_1k:  f64,
    output_per_1k: f64,
}

async fn handle_update_pricing(
    State(state): State<SharedState>,
    Json(body):   Json<PricingUpdate>,
) -> Json<Value> {
    state.budget.update_pricing(body.input_per_1k, body.output_per_1k).await;
    Json(json!({ "ok": true }))
}

async fn handle_spend(State(state): State<SharedState>) -> Json<Value> {
    let summary = state.budget.daily_summary(30).await.unwrap_or_default();
    Json(json!({ "daily": summary }))
}

// ── Federation ─────────────────────────────────────────────────────────────

async fn handle_federation_lookup(
    State(state): State<SharedState>,
    Path(hash):   Path<String>,
) -> Response {
    match state.cache.lookup_by_hash(&hash).await {
        Ok(Some(entry)) => {
            // Sign the response with our identity key so the requester can verify it
            let wire = entry_to_federated(&entry, &state.identity);
            Json(wire).into_response()
        }
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(e)   => {
            warn!("federation lookup error: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn handle_federation_announce(
    State(state): State<SharedState>,
    Json(payload): Json<AnnouncePayload>,
) -> Response {
    // 1. Verify self-signature — drop anything that fails
    if let Err(e) = payload.verify_self() {
        warn!("federation announce rejected (bad signature): {e}");
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid signature"}))).into_response();
    }

    // 2. Hard-reject evicted nodes immediately
    if state.trust.is_evicted(&payload.node_id).await {
        warn!("federation announce from evicted node {}", &payload.node_id[..16]);
        return StatusCode::FORBIDDEN.into_response();
    }

    // 3. Register the node (creates if new, updates last_seen if existing)
    let trust_state = match state.trust.register(&payload.node_id, &payload.public_key_hex, &payload.url).await {
        Ok(s)  => s,
        Err(e) => {
            warn!("trust register error: {e}");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    // 4. If a head-node counter-signed this node, try auto-promotion
    if let (Some(counter_by), Some(_counter_sig)) = (&payload.countersigned_by, &payload.counter_signature) {
        if let Ok(Some(head_key)) = state.trust.get_public_key(counter_by).await {
            if payload.verify_counter(&head_key).is_ok() {
                let _ = state.trust.auto_promote_if_head_signed(&payload.node_id, counter_by).await;
            }
        }
    }

    // 5. Untrusted nodes are acknowledged but their hashes are ignored
    if !trust_state.is_trusted() {
        info!(
            "federation announce from untrusted node {} ({} hashes ignored)",
            &payload.node_id[..16], payload.hashes.len()
        );
        return Json(json!({
            "ok":     true,
            "status": "untrusted",
            "note":   "node registered but not trusted; no cache entries will be fetched"
        })).into_response();
    }

    info!(
        "federation announce from trusted node {} ({} hashes)",
        &payload.node_id[..16], payload.hashes.len()
    );

    // Pull revocations from this peer in the background so we stay in sync
    {
        let fed   = state.federation.clone();
        let cache = state.cache.clone();
        let url   = payload.url.clone();
        tokio::spawn(async move {
            fed.pull_revocations_from_url(&url, &cache).await;
        });
    }

    Json(json!({ "ok": true, "status": "trusted", "received": payload.hashes.len() })).into_response()
}

async fn handle_federation_peers(State(state): State<SharedState>) -> Json<Value> {
    let trusted = state.trust.list_trusted().await.unwrap_or_default();
    Json(json!({
        "node_id":      state.node_id,
        "public_key":   state.identity.public_key_hex,
        "peer_count":   state.federation.peer_count(),
        "enabled":      state.federation.is_enabled(),
        "trusted_peers": trusted.len(),
    }))
}

// ── Trust management ────────────────────────────────────────────────────────

async fn handle_trust_list(State(state): State<SharedState>) -> Json<Value> {
    let nodes = state.trust.list_all().await.unwrap_or_default();
    Json(json!({ "nodes": nodes }))
}

#[derive(serde::Deserialize, Default)]
struct TrustPromoteBody {
    #[serde(default)]
    is_head: bool,
}

async fn handle_trust_promote(
    State(state): State<SharedState>,
    Path(node_id): Path<String>,
    body: Option<Json<TrustPromoteBody>>,
) -> Response {
    let is_head = body.map(|b| b.is_head).unwrap_or(false);
    match state.trust.promote(&node_id, &state.node_id, is_head).await {
        Ok(_) => Json(json!({ "ok": true, "node_id": node_id, "is_head": is_head })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct EvictBody {
    reason: String,
}

async fn handle_evict(
    State(state): State<SharedState>,
    Path(node_id): Path<String>,
    Json(body):    Json<EvictBody>,
) -> Response {
    // Sign the revocation with our own identity key
    let revocation_msg = crate::identity::revocation_message(&node_id, &body.reason);
    let sig = state.identity.sign(&revocation_msg);

    match state.trust.evict(&node_id, &body.reason, &state.node_id, &sig, &state.cache).await {
        Ok(_) => {
            // Push revocation to all trusted peers immediately
            let revocations = state.trust.list_revocations().await.unwrap_or_default();
            if let Some(rev) = revocations.iter().find(|r| r.node_id == node_id) {
                let fed   = state.federation.clone();
                let rev_c = rev.clone();
                tokio::spawn(async move { fed.push_revocation_to_peers(&rev_c).await });
            }
            Json(json!({ "ok": true, "node_id": node_id, "evicted": true })).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Revocation gossip endpoints ────────────────────────────────────────────

async fn handle_get_revocations(State(state): State<SharedState>) -> Json<Value> {
    let revocations = state.trust.list_revocations().await.unwrap_or_default();
    Json(json!({ "revocations": revocations }))
}

async fn handle_receive_revocation(
    State(state): State<SharedState>,
    Json(rev):    Json<crate::trust::RevocationRecord>,
) -> Response {
    match state.trust.apply_incoming_revocation(&rev, &state.cache).await {
        Ok(true) => {
            info!("applied incoming revocation for {}", &rev.node_id[..16.min(rev.node_id.len())]);
            // Do NOT re-push — one-hop push prevents broadcast storms.
            // Peers can pull from us via GET /v1/federation/revocations.
            Json(json!({ "ok": true, "applied": true })).into_response()
        }
        Ok(false) => Json(json!({ "ok": true, "applied": false })).into_response(),
        Err(e) => {
            warn!("revocation apply error: {e}");
            (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response()
        }
    }
}

// ── Passthrough fallback ───────────────────────────────────────────────────

async fn handle_passthrough(
    State(state): State<SharedState>,
    req:          axum::extract::Request,
) -> Response {
    let method  = req.method().clone();
    let uri     = req.uri().clone();
    let headers = req.headers().clone();
    let body    = axum::body::to_bytes(req.into_body(), usize::MAX).await.unwrap_or_default();

    let url = format!("{}{}", state.api_base_url.trim_end_matches('/'), uri.path_and_query().map(|p| p.as_str()).unwrap_or(""));

    let mut builder = reqwest::Client::new()
        .request(reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap(), &url);

    // Forward headers, replacing auth
    for (k, v) in headers.iter() {
        let name = k.as_str().to_lowercase();
        if name == "x-api-key" || name == "authorization" || name == "host" {
            continue;
        }
        if let Ok(val) = reqwest::header::HeaderValue::from_bytes(v.as_bytes()) {
            builder = builder.header(k.as_str(), val);
        }
    }

    // Inject auth
    builder = if state.api_creds.api_key.starts_with("sk-ant-oat") {
        builder.header("Authorization", format!("Bearer {}", state.api_creds.api_key))
    } else {
        builder.header("x-api-key", &state.api_creds.api_key)
    };

    builder = builder
        .header("anthropic-version", "2023-06-01")
        .body(body.to_vec());

    match builder.send().await {
        Ok(resp) => {
            let status  = resp.status();
            let rbytes  = resp.bytes().await.unwrap_or_default();
            Response::builder()
                .status(status.as_u16())
                .header("content-type", "application/json")
                .body(Body::from(rbytes))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(e) => {
            warn!("passthrough error: {e}");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}
