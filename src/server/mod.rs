use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router as AxumRouter,
};
use arc_swap::ArcSwap;
use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use std::num::NonZeroU32;
use bytes::Bytes;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{error, info, warn};

use crate::backend::{MessagesRequest, MessagesResponse};
use crate::budget::BudgetLedger;
use crate::cache::CacheStore;
use crate::config::AppConfig;
use crate::federation::{entry_to_federated, AnnouncePayload, FederationClient, PeerDescriptor, SemanticFederatedEntry, SemanticLookupRequest};
use crate::identity::announce_message;
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
    pub cfg:         Arc<ArcSwap<AppConfig>>,
    pub config_path: String,
    /// Ed25519 fingerprint — the canonical node identity used everywhere.
    pub node_id:          String,
    pub is_cnc:           bool,
    pub auto_promote_peers: bool,
    pub api_base_url:     String,
    pub api_creds:        crate::auth::Credentials,
    /// Static bearer token for the management portal.  `None` = auth disabled.
    /// Set via `CLAUDE_CACHE_PORTAL_TOKEN` env var — never stored in config files.
    pub portal_token:     Option<String>,
    /// Messages-per-minute rate limit applied to POST /v1/messages.  0 = disabled.
    pub rate_limit_rpm:   u32,
    /// Set when an Anthropic API call returns a credit-exhaustion error.
    /// While true, /v1/messages bypasses proxy routing and forwards with client credentials.
    pub credits_exhausted: AtomicBool,
}

pub type SharedState = Arc<AppState>;

// ── Portal auth middleware ─────────────────────────────────────────────────

async fn require_portal_token(
    State(state): State<SharedState>,
    request:      axum::extract::Request,
    next:         middleware::Next,
) -> Response {
    if let Some(token) = &state.portal_token {
        let ok = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|t| t == token.as_str())
            .unwrap_or(false);

        if !ok {
            return (
                StatusCode::UNAUTHORIZED,
                [("www-authenticate", "Bearer realm=\"claude-cache\"")],
                Json(json!({"error": "unauthorized"})),
            ).into_response();
        }
    }
    next.run(request).await
}

// ── Routes ─────────────────────────────────────────────────────────────────

pub fn build_router(state: SharedState) -> AxumRouter {
    // Protected routes require a portal token when one is configured.
    let protected = AxumRouter::new()
        .route("/stats",        get(handle_stats))
        .route("/api/pricing",       post(handle_update_pricing))
        .route("/api/spend",         get(handle_spend))
        .route("/api/credits/reset", post(handle_credits_reset))
        .route("/api/config/reload", post(handle_config_reload))
        .route("/v1/trust",              get(handle_trust_list))
        .route("/v1/trust/:node_id",     post(handle_trust_promote))
        .route("/v1/evict/:node_id",     post(handle_evict))
        .route("/",             get(crate::portal::handle_dashboard))
        .route("/api/overview", get(crate::portal::handle_overview))
        .route("/api/cache",    get(crate::portal::handle_cache_entries))
        .route("/api/trust",        get(crate::portal::handle_trust_nodes))
        .route("/api/peers/health", get(crate::portal::handle_peer_health))
        .route("/api/routing",      get(crate::portal::handle_routing_log))
        .route("/api/cache/search", get(crate::portal::handle_cache_search))
        .route("/v1/cache/export",             get(handle_export_cache))
        .route("/v1/cache/seed",               post(handle_seed_cache))
        .route("/v1/cache/entries/:id/pin",    post(handle_pin_cache_entry))
        .route("/v1/cache/entries/:id",        delete(handle_delete_cache_entry))
        .layer(middleware::from_fn_with_state(state.clone(), require_portal_token));

    // Rate limiter for /v1/messages — None when rate_limit_rpm == 0 (disabled).
    let messages_limiter: Option<Arc<DefaultDirectRateLimiter>> =
        NonZeroU32::new(state.rate_limit_rpm)
            .map(|rpm| Arc::new(RateLimiter::direct(Quota::per_minute(rpm))));

    // /v1/messages with optional rate-limiting middleware applied only to this route.
    let messages_router = {
        let lim = messages_limiter;
        AxumRouter::new()
            .route("/v1/messages", post(handle_messages))
            .layer(middleware::from_fn(move |req: axum::extract::Request, next: middleware::Next| {
                let lim = lim.clone();
                async move {
                    if let Some(ref limiter) = lim {
                        if limiter.check().is_err() {
                            return (
                                StatusCode::TOO_MANY_REQUESTS,
                                [(axum::http::header::RETRY_AFTER, "2")],
                                Json(json!({"error": "rate limit exceeded", "limit": "messages_per_minute"})),
                            ).into_response();
                        }
                    }
                    next.run(req).await
                }
            }))
    };

    // Remaining public routes: health check and federation peer endpoints
    // (federation uses Ed25519-based authentication, not portal token).
    let public = AxumRouter::new()
        .route("/health",                    get(handle_health))
        .route("/v1/federation/lookup/:hash", get(handle_federation_lookup))
        .route("/v1/federation/announce",    post(handle_federation_announce))
        .route("/v1/federation/peers",       get(handle_federation_peers))
        .route("/v1/federation/peers/list",  get(handle_federation_peers_list))
        .route("/v1/federation/semantic",    post(handle_federation_semantic))
        .route("/v1/federation/revocations", get(handle_get_revocations)
                                             .post(handle_receive_revocation))
        .fallback(handle_passthrough);

    AxumRouter::new()
        .merge(messages_router)
        .merge(public)
        .merge(protected)
        .with_state(state)
}

// ── /v1/messages ──────────────────────────────────────────────────────────

async fn handle_messages(
    State(state): State<SharedState>,
    headers:      HeaderMap,
    Json(mut req): Json<MessagesRequest>,
) -> Response {
    if let Some(beta) = headers.get("anthropic-beta").and_then(|v| v.to_str().ok()) {
        req.anthropic_beta = Some(beta.to_string());
    }

    let client_auth = extract_client_auth(&headers);

    if state.credits_exhausted.load(Ordering::Relaxed) {
        return handle_credit_bypass(&state, req, client_auth).await;
    }

    let is_stream = req.is_streaming();
    if is_stream {
        handle_stream_messages(state, req, client_auth).await
    } else {
        handle_sync_messages(state, req, client_auth).await
    }
}

async fn handle_sync_messages(
    state:       SharedState,
    req:         MessagesRequest,
    client_auth: Option<(String, String)>,
) -> Response {
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
        Err(e) if is_credit_exhausted(&e.to_string()) => {
            error!("API credit balance exhausted — proxy bypass activated. POST /api/credits/reset to restore after topping up.");
            state.credits_exhausted.store(true, Ordering::Relaxed);
            handle_credit_bypass(&state, req, client_auth).await
        }
        Err(e) => {
            warn!("router error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response()
        }
    }
}

async fn handle_stream_messages(
    state:       SharedState,
    req:         MessagesRequest,
    client_auth: Option<(String, String)>,
) -> Response {
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
        Err(e) if is_credit_exhausted(&e.to_string()) => {
            error!("API credit balance exhausted — proxy bypass activated. POST /api/credits/reset to restore after topping up.");
            state.credits_exhausted.store(true, Ordering::Relaxed);
            return handle_credit_bypass(&state, req, client_auth).await;
        }
        _ => {}
    }

    // True streaming passthrough to Anthropic
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let anthropic = state.anthropic.clone();
    let router    = state.router.clone();
    let req_clone = req.clone();

    tokio::spawn(async move {
        match anthropic.stream_to_channel(&req_clone, tx.clone()).await {
            Ok(acc) => {
                info!("stream complete: {} output tokens accumulated", acc.output_tokens);
                router.cache_streamed(&req_clone, &acc.text, &acc.message_id, acc.input_tokens, acc.output_tokens).await;
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
        "federation_peers": state.federation.peer_count().await,
        "credits_exhausted": state.credits_exhausted.load(Ordering::Relaxed),
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
        "credits_exhausted": state.credits_exhausted.load(Ordering::Relaxed),
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

async fn handle_config_reload(State(state): State<SharedState>) -> Response {
    match AppConfig::load(&state.config_path) {
        Ok(new_cfg) => {
            let old = state.cfg.load();
            if old.cache.db_path != new_cfg.cache.db_path
                || old.budget.db_path != new_cfg.budget.db_path
            {
                warn!("config reload: db_path changes ignored — restart required");
            }
            state.cfg.store(Arc::new(new_cfg));
            info!("config reloaded from {}", state.config_path);
            Json(json!({ "ok": true })).into_response()
        }
        Err(e) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": e.to_string() })),
        ).into_response(),
    }
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

    // 5a. CNC auto-promote: if we're a CNC configured to auto-promote, trust the
    //     node immediately so it gets a counter-signature in the response below.
    let trust_state = if state.is_cnc && state.auto_promote_peers && !trust_state.is_trusted() {
        info!("CNC auto-promoting {}", &payload.node_id[..16]);
        let _ = state.trust.promote(&payload.node_id, &state.node_id, false).await;
        state.trust.get_state(&payload.node_id).await.unwrap_or(trust_state)
    } else {
        trust_state
    };

    // 5b. Untrusted nodes are acknowledged but their hashes are ignored
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

    // 6. Process gossip peer list — register any unknown non-evicted peers as Untrusted
    if let Some(gossip_peers) = &payload.known_peers {
        for p in gossip_peers {
            if p.node_id == state.node_id || p.url.is_empty() { continue; }
            if state.trust.is_evicted(&p.node_id).await { continue; }
            if state.trust.is_trusted(&p.node_id).await { continue; }
            let _ = state.trust.register(&p.node_id, &p.public_key_hex, &p.url).await;
        }
    }

    // Pull revocations from this peer in the background so we stay in sync
    {
        let fed   = state.federation.clone();
        let cache = state.cache.clone();
        let url   = payload.url.clone();
        tokio::spawn(async move {
            fed.pull_revocations_from_url(&url, &cache).await;
        });
    }

    // If we are a CNC (head node), return a counter-signature so the client
    // can carry our endorsement to other peers for automatic promotion.
    if state.is_cnc {
        let counter_msg = announce_message(
            &payload.node_id,
            &payload.url,
            &payload.public_key_hex,
            &[],
        );
        let counter_sig = state.identity.sign(&counter_msg);
        return Json(json!({
            "ok":              true,
            "status":          "trusted",
            "received":        payload.hashes.len(),
            "counter_signature": counter_sig,
            "counter_node_id": state.identity.fingerprint,
        })).into_response();
    }

    Json(json!({ "ok": true, "status": "trusted", "received": payload.hashes.len() })).into_response()
}

async fn handle_federation_semantic(
    State(state): State<SharedState>,
    Json(req):    Json<SemanticLookupRequest>,
) -> Response {
    match state.cache.lookup_semantic(
        &req.domain,
        &req.embedding,
        req.sim_threshold,
        req.limit.min(10), // cap peer results at 10 to limit response size
    ).await {
        Ok(hits) => {
            let entries: Vec<SemanticFederatedEntry> = hits
                .into_iter()
                .map(|(entry, sim)| SemanticFederatedEntry {
                    entry:      entry_to_federated(&entry, &state.identity),
                    similarity: sim,
                })
                .collect();
            Json(entries).into_response()
        }
        Err(e) => {
            warn!("federation semantic lookup error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response()
        }
    }
}

async fn handle_federation_peers(State(state): State<SharedState>) -> Json<Value> {
    let trusted = state.trust.list_trusted().await.unwrap_or_default();
    Json(json!({
        "node_id":       state.identity.fingerprint,
        "public_key":    state.identity.public_key_hex,
        "is_cnc":        state.is_cnc,
        "peer_count":    state.federation.peer_count().await,
        "enabled":       state.federation.is_enabled(),
        "trusted_peers": trusted.len(),
    }))
}

/// Returns the list of trusted peers with their URLs and public keys.
/// Used by gossip discovery: new nodes call this to bootstrap knowledge
/// of the full mesh from a single known peer.
async fn handle_federation_peers_list(State(state): State<SharedState>) -> Json<Vec<PeerDescriptor>> {
    let peers = state.trust.list_trusted().await.unwrap_or_default();
    let list: Vec<PeerDescriptor> = peers
        .into_iter()
        .filter(|r| !r.url.is_empty() && r.node_id != state.node_id)
        .map(|r| PeerDescriptor {
            node_id:        r.node_id,
            url:            r.url,
            public_key_hex: r.public_key_hex,
        })
        .collect();
    Json(list)
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

// ── Cache management endpoints ─────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct SeedCacheBody {
    prompt:  String,
    response: String,
    #[serde(default)]
    system:  Option<String>,
    #[serde(default)]
    model:   Option<String>,
    #[serde(default)]
    domain:  Option<String>,
    #[serde(default)]
    pinned:  bool,
}

async fn handle_seed_cache(
    State(state): State<SharedState>,
    Json(body):   Json<SeedCacheBody>,
) -> Response {
    use crate::domain::{classify, ShapeKey};

    let shape = if let Some(ref d) = body.domain {
        ShapeKey { domain: d.clone(), intent: "generate".into(), complexity: 0.3 }
    } else {
        classify(&body.prompt)
    };

    let model = body.model.as_deref().unwrap_or("seeded");
    let ttl   = if body.pinned { None } else { Some(604_800u64) }; // 7 days for seeded entries

    match state.cache.store(
        &shape,
        &body.prompt,
        body.system.as_deref(),
        &body.response,
        model,
        None,
        ttl,
        false,
        body.pinned,
    ).await {
        Ok(id) => Json(serde_json::json!({ "ok": true, "id": id, "pinned": body.pinned })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct PinBody {
    #[serde(default = "bool_true")]
    pinned: bool,
}
fn bool_true() -> bool { true }

async fn handle_pin_cache_entry(
    State(state): State<SharedState>,
    Path(id):     Path<String>,
    body:         Option<Json<PinBody>>,
) -> Response {
    let pinned = body.map(|b| b.pinned).unwrap_or(true);
    match state.cache.set_pinned(&id, pinned).await {
        Ok(true)  => Json(serde_json::json!({ "ok": true, "id": id, "pinned": pinned })).into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "entry not found"}))).into_response(),
        Err(e)    => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

async fn handle_delete_cache_entry(
    State(state): State<SharedState>,
    Path(id):     Path<String>,
) -> Response {
    match state.cache.delete_entry(&id).await {
        Ok(true)  => Json(serde_json::json!({ "ok": true, "id": id })).into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "entry not found"}))).into_response(),
        Err(e)    => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Cache export ──────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct ExportParams {
    #[serde(default)]
    domain:  Option<String>,
    #[serde(default)]
    pinned:  Option<bool>,
    #[serde(default)]
    limit:   Option<i64>,
}

async fn handle_export_cache(
    State(state): State<SharedState>,
    Query(params): Query<ExportParams>,
) -> Response {
    let limit       = params.limit.unwrap_or(1000).min(5000);
    let pinned_only = params.pinned.unwrap_or(false);

    match state.cache.export_entries(params.domain.as_deref(), pinned_only, limit).await {
        Ok(entries) => {
            let body = match serde_json::to_vec(&entries) {
                Ok(b) => b,
                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()}))).into_response(),
            };
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "application/json")
                .header("content-disposition", "attachment; filename=\"cache-export.json\"")
                .body(axum::body::Body::from(body))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Credit exhaustion helpers ─────────────────────────────────────────────

fn is_credit_exhausted(msg: &str) -> bool {
    msg.contains("credit balance is too low")
}

fn extract_client_auth(headers: &HeaderMap) -> Option<(String, String)> {
    if let Some(v) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        return Some(("Authorization".to_string(), v.to_string()));
    }
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(("x-api-key".to_string(), v.to_string()));
    }
    None
}

/// Forward a request directly to the Anthropic API using the client's own credentials,
/// bypassing all proxy routing and caching.  Used when the proxy's API credit balance
/// is exhausted but the client (e.g. Claude Code) has its own OAuth credits.
async fn handle_credit_bypass(
    state:       &SharedState,
    req:         MessagesRequest,
    client_auth: Option<(String, String)>,
) -> Response {
    let Some((auth_name, auth_val)) = client_auth else {
        warn!("credit bypass: no client credentials in request — cannot forward");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "type":  "error",
                "error": {
                    "type":    "credit_exhausted",
                    "message": "API credit balance exhausted and no client credentials available for bypass. POST /api/credits/reset after topping up."
                }
            })),
        ).into_response();
    };

    let is_stream = req.is_streaming();
    warn!("credit bypass: forwarding {} to Anthropic with client credentials (caching skipped)",
        if is_stream { "stream" } else { "request" });

    let url = format!("{}/v1/messages", state.api_base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let mut builder = client
        .post(&url)
        .header(auth_name.as_str(), auth_val.as_str())
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json");

    if let Some(beta) = &req.anthropic_beta {
        builder = builder.header("anthropic-beta", beta.as_str());
    }

    match builder.json(&req).send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            if is_stream {
                let body = Body::from_stream(resp.bytes_stream());
                Response::builder()
                    .status(status)
                    .header("content-type", "text/event-stream")
                    .header("cache-control", "no-cache")
                    .header("x-router-source", "credit-bypass-stream")
                    .body(body)
                    .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
            } else {
                let bytes = resp.bytes().await.unwrap_or_default();
                Response::builder()
                    .status(status)
                    .header("content-type", "application/json")
                    .header("x-router-source", "credit-bypass")
                    .body(Body::from(bytes))
                    .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
            }
        }
        Err(e) => {
            warn!("credit bypass request failed: {e}");
            (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("credit bypass failed: {e}")}))).into_response()
        }
    }
}

async fn handle_credits_reset(State(state): State<SharedState>) -> Json<Value> {
    let was_exhausted = state.credits_exhausted.swap(false, Ordering::Relaxed);
    if was_exhausted {
        info!("credit exhaustion flag cleared — proxy routing restored");
    }
    Json(json!({ "ok": true, "was_exhausted": was_exhausted, "proxy_mode": "restored" }))
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
