use std::sync::Arc;
use anyhow::Result;
use tracing::info;

use claude_cache::{
    auth,
    backend::{AnthropicBackend, ModelBackend, OllamaBackend},
    budget::BudgetLedger,
    cache::CacheStore,
    config::AppConfig,
    embedding::{Embedder, OllamaEmbedder, StubEmbedder},
    federation::{FederationClient, peers_from_urls},
    identity::NodeIdentity,
    router::Router,
    server::{AppState, build_router},
    trust::TrustStore,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "claude_cache=info,tower_http=warn".into()),
        )
        .init();

    let cfg = Arc::new(AppConfig::load("config.toml").unwrap_or_else(|e| {
        tracing::warn!("config.toml not found ({e}), using defaults");
        AppConfig::default()
    }));

    // ── Node identity ─────────────────────────────────────────────────────
    let identity = Arc::new(NodeIdentity::load_or_generate("node_identity.key")?);
    info!("node identity: {}", &identity.fingerprint[..16]);
    info!("public key:    {}", identity.public_key_hex);

    // ── Credentials ───────────────────────────────────────────────────────
    let creds = auth::load()?;

    // ── Stores ────────────────────────────────────────────────────────────
    let cache  = Arc::new(CacheStore::open(&cfg.cache.db_path, &identity.fingerprint).await?);
    let budget = Arc::new(BudgetLedger::open(cfg.budget.clone()).await?);
    let trust  = Arc::new(TrustStore::open("claude-cache.trust.db", &identity.fingerprint).await?);

    info!("listening on {}:{}", cfg.server.host, cfg.server.port);

    // ── Embedder ──────────────────────────────────────────────────────────
    let embedder: Arc<dyn Embedder> = if cfg.embedding.enabled {
        Arc::new(OllamaEmbedder::new(&cfg.embedding))
    } else {
        Arc::new(StubEmbedder::new(cfg.embedding.dimensions))
    };

    // ── Backends ──────────────────────────────────────────────────────────
    let anthropic = Arc::new(AnthropicBackend::new(&cfg.api, creds.clone()));
    let ollama    = Arc::new(OllamaBackend::new(&cfg.local));

    let local_backend: Arc<dyn ModelBackend> = ollama;
    let api_backend:   Arc<dyn ModelBackend> = anthropic.clone();

    // ── Federation ────────────────────────────────────────────────────────
    let peers = peers_from_urls(&cfg.federation.peers);
    let federation = Arc::new(FederationClient::new(
        peers,
        cfg.federation.enabled,
        identity.clone(),
        trust.clone(),
        cfg.federation.lookup_timeout_ms,
    ));

    // ── Router ────────────────────────────────────────────────────────────
    let router = Router::new(
        cfg.clone(),
        cache.clone(),
        budget.clone(),
        embedder,
        local_backend,
        api_backend,
    );

    // ── Startup: pull revocations from known peers ────────────────────────
    {
        let fed_sync   = federation.clone();
        let cache_sync = cache.clone();
        tokio::spawn(async move {
            let applied = fed_sync.sync_revocations(&cache_sync).await;
            if applied > 0 { info!("startup: applied {applied} revocations from peers"); }
        });
    }

    // ── Background: eviction + gossip + revocation sync ───────────────────
    {
        let evict_cache = cache.clone();
        let fed         = federation.clone();
        let our_url     = format!("http://{}:{}", cfg.server.host, cfg.server.port);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
            loop {
                interval.tick().await;
                if let Ok(n) = evict_cache.evict_expired().await {
                    if n > 0 { info!("evicted {n} expired cache entries"); }
                }
                if let Ok(hashes) = evict_cache.list_shared_hashes(500, 0).await {
                    fed.announce(hashes, &our_url).await;
                }
                let applied = fed.sync_revocations(&evict_cache).await;
                if applied > 0 { info!("hourly sync: applied {applied} revocations from peers"); }
            }
        });
    }

    // ── Server ────────────────────────────────────────────────────────────
    let state = Arc::new(AppState {
        router,
        cache,
        budget,
        federation,
        trust,
        identity,
        anthropic,
        node_id:      cfg.server.node_id.clone(),
        api_base_url: cfg.api.base_url.clone(),
        api_creds:    creds,
    });

    let app      = build_router(state);
    let addr     = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("dashboard:  http://{addr}");
    info!("trust mgmt: POST http://{addr}/v1/trust/:node_id");
    info!("evict:      POST http://{addr}/v1/evict/:node_id");

    axum::serve(listener, app).await?;
    Ok(())
}
