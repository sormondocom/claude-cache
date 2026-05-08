use std::sync::Arc;
use anyhow::Result;
use arc_swap::ArcSwap;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tracing::info;

use claude_cache::{
    auth,
    backend::{AnthropicBackend, ModelBackend, OllamaBackend},
    budget::BudgetLedger,
    cache::CacheStore,
    config::{AppConfig, NodeRole},
    embedding::{Embedder, OllamaEmbedder, StubEmbedder},
    federation::{AnnouncePayload, FederationClient},
    health,
    identity::NodeIdentity,
    router::Router,
    server::{AppState, build_router},
    trust::TrustStore,
};

// ── CLI ────────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "claude-cache",
    about   = "Anthropic API cache and routing proxy",
    version
)]
struct Cli {
    /// Path to config file
    #[arg(long, default_value = "config.toml", global = true)]
    config: String,

    /// Override node role: cnc or client
    #[arg(long)]
    role: Option<String>,

    /// CNC URL for client boot-strapping (overrides config)
    #[arg(long)]
    cnc_url: Option<String>,

    /// CNC Ed25519 fingerprint for client bootstrapping (overrides config)
    #[arg(long)]
    cnc_node_id: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Print this node's stable Ed25519 fingerprint and public key, then exit.
    /// Run this on any node to get the value to put in a peer's config.toml.
    Identity {
        /// Path to key file (default: node_identity.key)
        #[arg(long, default_value = "node_identity.key")]
        key_file: String,
    },
}

// ── Counter-signature persistence ──────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct CounterSigFile {
    counter_node_id:  String,
    counter_signature: String,
}

fn load_countersig(path: &str) -> (Option<String>, Option<String>) {
    let Ok(text) = std::fs::read_to_string(path) else { return (None, None); };
    let Ok(csf)  = serde_json::from_str::<CounterSigFile>(&text) else { return (None, None); };
    (Some(csf.counter_node_id), Some(csf.counter_signature))
}

fn save_countersig(path: &str, counter_node_id: &str, counter_signature: &str) {
    let csf = CounterSigFile {
        counter_node_id:   counter_node_id.to_string(),
        counter_signature: counter_signature.to_string(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&csf) {
        let _ = std::fs::write(path, json);
    }
}

// ── Entry point ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Identity { key_file }) => {
            let id = NodeIdentity::load_or_generate(&key_file)?;
            println!("fingerprint: {}", id.fingerprint);
            println!("public_key:  {}", id.public_key_hex);
            return Ok(());
        }
        None => {
            run_server(cli.config, cli.role, cli.cnc_url, cli.cnc_node_id).await
        }
    }
}

// ── Server ─────────────────────────────────────────────────────────────────────

async fn run_server(
    config_path: String,
    role_override: Option<String>,
    cnc_url_override: Option<String>,
    cnc_node_id_override: Option<String>,
) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "claude_cache=info,tower_http=warn".into()),
        )
        .init();

    let mut cfg = AppConfig::load(&config_path).unwrap_or_else(|e| {
        tracing::warn!("config.toml not found ({e}), using defaults");
        AppConfig::default()
    });

    // Apply CLI overrides to node config
    if let Some(role_str) = role_override {
        cfg.node.role = match role_str.to_lowercase().as_str() {
            "cnc" => NodeRole::Cnc,
            _     => NodeRole::Client,
        };
    }
    if let Some(url) = cnc_url_override     { cfg.node.cnc_url     = url; }
    if let Some(nid) = cnc_node_id_override { cfg.node.cnc_node_id = nid; }

    let cfg = Arc::new(ArcSwap::new(Arc::new(cfg)));
    let is_cnc = cfg.load().node.role == NodeRole::Cnc;

    // ── Node identity ─────────────────────────────────────────────────────
    let identity = Arc::new(NodeIdentity::load_or_generate("node_identity.key")?);
    info!("node fingerprint: {}", &identity.fingerprint[..16]);
    info!("public key:       {}", identity.public_key_hex);
    if is_cnc { info!("role: CNC (head node)"); } else { info!("role: client"); }

    // ── Credentials ───────────────────────────────────────────────────────
    let creds = auth::load()?;

    // Snapshot for startup use — these fields seed one-time-init structs and
    // are safe to read once; the live config is accessed via cfg.load() later.
    let c = cfg.load();

    // ── Stores ────────────────────────────────────────────────────────────
    let cache  = Arc::new(CacheStore::open(&c.cache.db_path, &identity.fingerprint).await?);
    let budget = Arc::new(BudgetLedger::open(c.budget.clone()).await?);
    let trust  = Arc::new(TrustStore::open("claude-cache.trust.db", &identity.fingerprint).await?);

    let our_url = format!("http://{}:{}", c.server.host, c.server.port);

    // ── Bootstrap trust for config-declared peers ─────────────────────────
    // Config peers are explicitly trusted: operator declared them.
    for peer in &c.federation.peers {
        trust.register_config_peer(
            &peer.node_id,
            &peer.public_key_hex,
            &peer.url,
            false,
        ).await?;
        info!("config peer trusted: {} ({})", &peer.node_id[..16.min(peer.node_id.len())], peer.url);
    }

    // ── CNC: register self as head node ───────────────────────────────────
    if is_cnc {
        trust.register_config_peer(
            &identity.fingerprint,
            &identity.public_key_hex,
            &our_url,
            true,
        ).await?;
        info!("registered self as head node");
    }

    // ── Client: register + trust the CNC if configured ───────────────────
    let cnc_url_resolved = if !c.node.cnc_url.is_empty() {
        Some(c.node.cnc_url.clone())
    } else {
        None
    };

    if let Some(ref cnc_url) = cnc_url_resolved {
        if !c.node.cnc_node_id.is_empty() {
            // Trust the CNC immediately — we know its fingerprint from config
            trust.register_config_peer(
                &c.node.cnc_node_id,
                "",   // public key filled in on first announce from CNC
                cnc_url,
                true, // CNC is a head node
            ).await?;
            info!("CNC trusted: {} ({})", &c.node.cnc_node_id[..16.min(c.node.cnc_node_id.len())], cnc_url);
        }
    }

    info!("listening on {}:{}", c.server.host, c.server.port);

    // ── Embedder ──────────────────────────────────────────────────────────
    let embedder: Arc<dyn Embedder> = if c.embedding.enabled {
        Arc::new(OllamaEmbedder::new(&c.embedding))
    } else {
        Arc::new(StubEmbedder::new(c.embedding.dimensions))
    };

    // ── Backends ──────────────────────────────────────────────────────────
    let anthropic = Arc::new(AnthropicBackend::new(&c.api, creds.clone()));
    let ollama    = Arc::new(OllamaBackend::new(&c.local));

    let local_backend: Arc<dyn ModelBackend> = ollama;
    let api_backend:   Arc<dyn ModelBackend> = anthropic.clone();

    // ── Federation ────────────────────────────────────────────────────────
    let federation = Arc::new(FederationClient::new(
        c.federation.enabled,
        identity.clone(),
        trust.clone(),
        c.federation.lookup_timeout_ms,
    ));

    // ── Router ────────────────────────────────────────────────────────────
    let router = Router::new(
        cfg.clone(),
        cache.clone(),
        budget.clone(),
        embedder,
        local_backend,
        api_backend,
    ).with_federation(federation.clone());

    // ── Startup: pull revocations from known peers ────────────────────────
    {
        let fed_sync   = federation.clone();
        let cache_sync = cache.clone();
        tokio::spawn(async move {
            let applied = fed_sync.sync_revocations(&cache_sync).await;
            if applied > 0 { info!("startup: applied {applied} revocations from peers"); }
        });
    }

    // ── Startup: bootstrap peer discovery from config peers ───────────────
    // Pull each config peer's trusted peer list so we learn the full mesh
    // after a single hop without needing everyone in config.toml.
    if c.federation.enabled && !c.federation.peers.is_empty() {
        let fed_boot  = federation.clone();
        let peer_urls: Vec<String> = c.federation.peers.iter().map(|p| p.url.clone()).collect();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            for url in &peer_urls {
                fed_boot.exchange_peers(url).await;
            }
        });
    }

    // ── Client bootstrap: announce to CNC ────────────────────────────────
    if let Some(cnc_url) = cnc_url_resolved.clone() {
        let identity_b = identity.clone();
        let cache_b    = cache.clone();
        let our_url_b  = our_url.clone();
        tokio::spawn(async move {
            // Small delay — let the server start binding before CNC tries to
            // contact us back.
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;

            let hashes = cache_b.list_shared_hashes(500, 0).await.unwrap_or_default();

            // Attach any stored counter-signature from a previous CNC bootstrap
            let (countersigned_by, counter_signature) = load_countersig("node_countersig.json");

            let mut payload = AnnouncePayload::build(&identity_b, &our_url_b, hashes);
            payload.countersigned_by  = countersigned_by;
            payload.counter_signature = counter_signature;

            let client = reqwest::Client::new();
            match client
                .post(format!("{cnc_url}/v1/federation/announce"))
                .json(&payload)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(body) = resp.json::<serde_json::Value>().await {
                        // If the CNC returned a counter-signature, persist it so
                        // future announces to other peers carry CNC endorsement.
                        if let (Some(sig), Some(by)) = (
                            body.get("counter_signature").and_then(|v| v.as_str()),
                            body.get("counter_node_id").and_then(|v| v.as_str()),
                        ) {
                            save_countersig("node_countersig.json", by, sig);
                            info!("received counter-signature from CNC {}", &by[..16.min(by.len())]);
                        }
                        info!("CNC bootstrap: {}", body.get("status").and_then(|v| v.as_str()).unwrap_or("ok"));
                    }
                }
                Ok(resp) => {
                    tracing::warn!("CNC bootstrap rejected: HTTP {}", resp.status());
                }
                Err(e) => {
                    tracing::warn!("CNC bootstrap failed (server may not be up yet): {e}");
                }
            }
        });
    }

    // ── Background: peer health checks ───────────────────────────────────
    if c.health.enabled && c.federation.enabled {
        let health_trust  = trust.clone();
        let health_cfg    = c.health.clone();
        let health_our_id = identity.fingerprint.clone();
        tokio::spawn(async move {
            health::run(health_trust, health_cfg, health_our_id).await;
        });
    }

    // ── Background: eviction + gossip + revocation sync ───────────────────
    {
        let evict_cache    = cache.clone();
        let fed            = federation.clone();
        let trust_bg       = trust.clone();
        let bg_url         = our_url.clone();
        let max_size_bytes = c.cache.max_size_mb * 1024 * 1024;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
            loop {
                interval.tick().await;
                if let Ok(n) = evict_cache.evict_expired().await {
                    if n > 0 { info!("evicted {n} expired cache entries"); }
                }
                match evict_cache.evict_to_size_limit(max_size_bytes).await {
                    Ok(n) if n > 0 => info!("size-limit eviction: removed {n} LRU entries"),
                    Err(e)         => tracing::warn!("size-limit eviction error: {e}"),
                    _              => {}
                }
                if let Ok(hashes) = evict_cache.list_shared_hashes(500, 0).await {
                    fed.announce(hashes, &bg_url).await;
                }
                // Pull peer lists from all trusted peers to keep discovery fresh.
                if let Ok(known) = trust_bg.list_trusted().await {
                    for peer in known {
                        if !peer.url.is_empty() {
                            fed.exchange_peers(&peer.url).await;
                        }
                    }
                }
                let applied = fed.sync_revocations(&evict_cache).await;
                if applied > 0 { info!("hourly sync: applied {applied} revocations from peers"); }
            }
        });
    }

    // ── Portal token (env only — never in config.toml) ────────────────────
    let portal_token = std::env::var("CLAUDE_CACHE_PORTAL_TOKEN").ok()
        .filter(|t| !t.is_empty());
    if portal_token.is_some() {
        info!("portal auth: enabled (CLAUDE_CACHE_PORTAL_TOKEN is set)");
    } else {
        info!("portal auth: disabled (set CLAUDE_CACHE_PORTAL_TOKEN to enable)");
    }

    // ── Background: config file mtime watcher (auto hot-reload) ──────────
    {
        let watch_path = config_path.clone();
        let watch_cfg  = cfg.clone();
        tokio::spawn(async move {
            let mut last_mtime = std::fs::metadata(&watch_path).ok()
                .and_then(|m| m.modified().ok());
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
            loop {
                interval.tick().await;
                let current = std::fs::metadata(&watch_path).ok()
                    .and_then(|m| m.modified().ok());
                if current.is_some() && current != last_mtime {
                    last_mtime = current;
                    match AppConfig::load(&watch_path) {
                        Ok(new_cfg) => {
                            watch_cfg.store(Arc::new(new_cfg));
                            info!("config auto-reloaded from {watch_path}");
                        }
                        Err(e) => tracing::warn!("config auto-reload failed: {e}"),
                    }
                }
            }
        });
    }

    // ── Server ────────────────────────────────────────────────────────────
    let node_id    = identity.fingerprint.clone();
    let rate_limit = c.limits.messages_per_minute;
    let api_url    = c.api.base_url.clone();
    let auto_promo = c.node.auto_promote_peers;
    let drain_secs = c.limits.shutdown_timeout_secs;
    let fed_enabled = c.federation.enabled;
    let rpm_limit   = c.limits.messages_per_minute;
    // Drop the startup snapshot — AppState holds cfg for runtime reads.
    drop(c);

    let state = Arc::new(AppState {
        router,
        cache,
        budget,
        federation,
        trust,
        identity,
        anthropic,
        cfg:         cfg.clone(),
        config_path: config_path.clone(),
        node_id,
        is_cnc,
        auto_promote_peers:  auto_promo,
        api_base_url:        api_url,
        api_creds:           creds,
        portal_token,
        rate_limit_rpm:      rate_limit,
    });

    let app      = build_router(state);
    let addr     = format!("{}:{}", cfg.load().server.host, cfg.load().server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("─── endpoints ───────────────────────────────────────────");
    info!("  POST  http://{addr}/v1/messages          (proxy)");
    info!("  GET   http://{addr}/health");
    if fed_enabled {
        info!("  POST  http://{addr}/v1/federation/announce");
        info!("  GET   http://{addr}/v1/federation/peers");
        info!("  GET   http://{addr}/v1/federation/lookup/:hash");
        info!("  POST  http://{addr}/v1/federation/semantic");
        info!("  GET   http://{addr}/v1/federation/revocations");
        info!("  POST  http://{addr}/v1/federation/revocations");
    }
    info!("─── portal (protected) ──────────────────────────────────");
    info!("  GET   http://{addr}/                     (dashboard)");
    info!("  GET   http://{addr}/stats");
    info!("  GET   http://{addr}/api/overview");
    info!("  GET   http://{addr}/api/cache");
    info!("  GET   http://{addr}/api/cache/search");
    info!("  GET   http://{addr}/api/spend");
    info!("  POST  http://{addr}/api/pricing");
    info!("  POST  http://{addr}/api/config/reload");
    info!("  GET   http://{addr}/api/trust");
    info!("  GET   http://{addr}/api/peers/health");
    info!("  GET   http://{addr}/api/routing");
    info!("─── cache management ────────────────────────────────────");
    info!("  GET   http://{addr}/v1/cache/export");
    info!("  POST  http://{addr}/v1/cache/seed");
    info!("  POST  http://{addr}/v1/cache/entries/:id/pin");
    info!("  DELETE http://{addr}/v1/cache/entries/:id");
    if is_cnc {
        info!("─── trust / eviction (CNC) ──────────────────────────────");
        info!("  GET   http://{addr}/v1/trust");
        info!("  POST  http://{addr}/v1/trust/:node_id");
        info!("  POST  http://{addr}/v1/evict/:node_id");
    }
    info!("─────────────────────────────────────────────────────────");
    if rpm_limit > 0 {
        info!("rate limit: {} req/min on POST /v1/messages", rpm_limit);
    }

    // Wait for SIGTERM/Ctrl+C, then give in-flight requests up to drain_secs
    // to complete before forcing exit.  The sleep starts AFTER the signal so
    // the node runs indefinitely until explicitly stopped.
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            info!("shutting down — draining in-flight requests (up to {drain_secs}s)");
            tokio::time::sleep(std::time::Duration::from_secs(drain_secs)).await;
        })
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c    => { tracing::info!("received Ctrl+C — stopping"); }
        _ = terminate => { tracing::info!("received SIGTERM — stopping"); }
    }
}
