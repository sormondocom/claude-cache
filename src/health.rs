/// Background peer health checker.
///
/// Runs at a configurable interval, probing every trusted peer's /health
/// endpoint.  Results are stored in TrustStore.peer_health so the federation
/// client can skip unreachable peers and prefer lower-latency ones.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::config::HealthConfig;
use crate::trust::TrustStore;

pub async fn run(trust: Arc<TrustStore>, cfg: HealthConfig, our_node_id: String) {
    if !cfg.enabled {
        debug!("health checker disabled");
        return;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(cfg.timeout_ms))
        .use_rustls_tls()
        .build()
        .expect("health check client");

    info!(
        "health checker started: interval={}s timeout={}ms failure_threshold={}",
        cfg.interval_secs, cfg.timeout_ms, cfg.failure_threshold
    );

    let mut interval = tokio::time::interval(Duration::from_secs(cfg.interval_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    interval.tick().await; // skip the immediate first tick

    loop {
        interval.tick().await;

        let peers = match trust.list_trusted().await {
            Ok(p)  => p,
            Err(e) => { warn!("health check: failed to list trusted peers: {e}"); continue; }
        };

        let mut checked = 0usize;
        for peer in &peers {
            if peer.node_id == our_node_id || peer.url.is_empty() {
                continue;
            }

            let url = format!("{}/health", peer.url.trim_end_matches('/'));
            let start = Instant::now();

            let result = client.get(&url).send().await;
            let elapsed_ms = start.elapsed().as_millis() as u64;

            let (success, latency) = match result {
                Ok(resp) if resp.status().is_success() => (true, Some(elapsed_ms)),
                Ok(resp) => {
                    debug!("peer {} health: HTTP {}", &peer.node_id[..16.min(peer.node_id.len())], resp.status());
                    (false, None)
                }
                Err(e) => {
                    debug!("peer {} unreachable: {e}", &peer.node_id[..16.min(peer.node_id.len())]);
                    (false, None)
                }
            };

            if let Err(e) = trust.record_health_check(
                &peer.node_id, &peer.url, success, latency, cfg.failure_threshold,
            ).await {
                warn!("health record error: {e}");
            }

            if success {
                debug!(
                    "peer {} ok: {}ms", &peer.node_id[..16.min(peer.node_id.len())], elapsed_ms
                );
            }
            checked += 1;
        }

        if checked > 0 {
            debug!("health cycle complete: checked {} peers", checked);
        }
    }
}
