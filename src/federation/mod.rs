/// Federation layer — content-addressed cache sharing across nodes.
///
/// Security model:
///   - Every announce is signed with the sender's Ed25519 key.
///   - Peers verify the signature before processing any announce.
///   - Only TRUSTED peers are queried for cache lookups.
///   - Untrusted peers are registered but never contribute to the cache.
///   - Evicted peers are dropped immediately; their cached entries are purged.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::cache::{CacheEntry, CacheStore};
use crate::identity::{announce_message, NodeIdentity, RemoteKey};
use crate::trust::{RevocationRecord, TrustStore};

// ── Wire types ─────────────────────────────────────────────────────────────

/// Wire type returned by /v1/federation/lookup/:hash
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedEntry {
    pub hash:       String,
    pub response:   String,
    pub model_used: String,
    pub domain:     String,
    pub node_id:    String,
    /// Signature over: sha256(hash + response + node_id) — hex Ed25519 sig
    pub signature:  String,
}

/// Request body for POST /v1/federation/semantic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticLookupRequest {
    pub domain:        String,
    pub embedding:     Vec<f32>,
    pub sim_threshold: f64,
    pub limit:         usize,
}

/// Single result from POST /v1/federation/semantic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFederatedEntry {
    pub entry:      FederatedEntry,
    pub similarity: f64,
}

/// Wire type for /v1/federation/announce
#[derive(Debug, Serialize, Deserialize)]
pub struct AnnouncePayload {
    pub node_id:       String,
    pub url:           String,
    pub public_key_hex: String,
    /// Optional: node_id of a head-node that has countersigned this key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub countersigned_by: Option<String>,
    /// Optional: head-node's signature over `announce_message(node_id, url, public_key_hex, [])`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counter_signature: Option<String>,
    pub hashes:        Vec<String>,
    /// Ed25519 signature over `announce_message(node_id, url, public_key_hex, hashes)`
    pub signature:     String,
    /// Optional gossip: trusted peers we know about, forwarded so the recipient
    /// can discover the mesh without static config.  Not covered by the signature
    /// (advisory only — recipients validate each entry independently).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub known_peers: Option<Vec<PeerDescriptor>>,
}

impl AnnouncePayload {
    pub fn build(identity: &NodeIdentity, url: &str, hashes: Vec<String>) -> Self {
        let msg = announce_message(&identity.fingerprint, url, &identity.public_key_hex, &hashes);
        let sig = identity.sign(&msg);
        AnnouncePayload {
            node_id:          identity.fingerprint.clone(),
            url:              url.to_string(),
            public_key_hex:   identity.public_key_hex.clone(),
            countersigned_by: None,
            counter_signature: None,
            hashes,
            signature:        sig,
            known_peers:      None,
        }
    }

    /// Verify the self-signature.  Returns Err if the signature is invalid.
    pub fn verify_self(&self) -> anyhow::Result<()> {
        let key = RemoteKey::from_hex(&self.public_key_hex)?;
        // The node_id must match the fingerprint of the claimed public key
        let expected_fp = crate::identity::fingerprint_of(&hex::decode(&self.public_key_hex)?);
        if expected_fp != self.node_id {
            anyhow::bail!("announce node_id does not match public key fingerprint");
        }
        let msg = announce_message(&self.node_id, &self.url, &self.public_key_hex, &self.hashes);
        key.verify(&msg, &self.signature)
    }

    /// Verify the optional head-node counter-signature.
    pub fn verify_counter(&self, head_key: &RemoteKey) -> anyhow::Result<()> {
        let sig = self.counter_signature.as_deref()
            .ok_or_else(|| anyhow::anyhow!("no counter-signature present"))?;
        // The head node signed: announce_message(node_id, url, public_key_hex, [])
        let msg = announce_message(&self.node_id, &self.url, &self.public_key_hex, &[]);
        head_key.verify(&msg, sig)
    }
}

/// Signed federation lookup response.
#[derive(Debug, Serialize, Deserialize)]
pub struct LookupResponse {
    pub entry:    FederatedEntry,
    pub node_id:  String,
}

// ── Peer node descriptor ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerNode {
    pub id:  String,
    pub url: String,
}

/// Domain knowledge payload served by `GET /v1/federation/knowledge/:domain`.
/// Carries the local node's distilled knowledge doc, per-intent calibration biases,
/// and labeled contrast pairs so peers can incorporate mesh-wide learning.
#[derive(Debug, Serialize, Deserialize)]
pub struct PeerKnowledge {
    pub domain:             String,
    pub node_id:            String,
    pub knowledge_doc:      Option<String>,
    /// `intent → bias` — additive correction to local model confidence for this domain.
    pub calibration_biases: std::collections::HashMap<String, f64>,
    pub contrast_pairs:     Vec<PeerContrastPair>,
}

/// A labeled failure case from a peer node: local_attempt was wrong, correct
/// is the API-authoritative answer.  Peers incorporate these into distillation
/// so the mesh collectively learns what patterns to avoid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerContrastPair {
    pub intent:  String,
    pub wrong:   String,
    pub correct: String,
}

/// Minimal peer advertisement used for gossip discovery.
/// Carried in AnnouncePayload.known_peers and returned by
/// GET /v1/federation/peers/list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerDescriptor {
    pub node_id:        String,
    pub url:            String,
    pub public_key_hex: String,
}

// ── Client ─────────────────────────────────────────────────────────────────

pub struct FederationClient {
    client:   reqwest::Client,
    enabled:  bool,
    identity: Arc<NodeIdentity>,
    trust:    Arc<TrustStore>,
}

impl FederationClient {
    pub fn new(
        enabled:    bool,
        identity:   Arc<NodeIdentity>,
        trust:      Arc<TrustStore>,
        timeout_ms: u64,
    ) -> Self {
        FederationClient {
            client: reqwest::Client::builder()
                .timeout(Duration::from_millis(timeout_ms))
                .use_rustls_tls()
                .build()
                .expect("federation reqwest client"),
            enabled,
            identity,
            trust,
        }
    }

    /// All trusted peers with a known URL, excluding this node.
    /// Derives the live set from TrustStore on every call, so config-declared
    /// peers and dynamically-promoted peers are both included.
    async fn live_peers(&self) -> Vec<PeerNode> {
        let own_id = &self.identity.fingerprint;
        self.trust.list_trusted().await.unwrap_or_default()
            .into_iter()
            .filter(|r| !r.url.is_empty() && &r.node_id != own_id)
            .map(|r| PeerNode { id: r.node_id, url: r.url })
            .collect()
    }

    /// Query TRUSTED, REACHABLE peers for a cache entry by SHA256 hash.
    /// Peers are sorted by average latency (fastest first) before the parallel
    /// fan-out, so the first result to resolve is most likely to be the fastest.
    pub async fn lookup(&self, hash: &str) -> Option<FederatedEntry> {
        if !self.enabled { return None; }

        let peers = self.reachable_peers_sorted().await;
        if peers.is_empty() { return None; }

        let mut handles = Vec::new();
        for peer in peers {
            let client  = self.client.clone();
            let trust   = self.trust.clone();
            let url     = format!("{}/v1/federation/lookup/{}", peer.url, hash);
            let peer_id = peer.id.clone();
            handles.push(tokio::spawn(async move {
                let resp = client.get(&url).send().await.ok()?;
                if !resp.status().is_success() { return None; }
                let entry: FederatedEntry = resp.json().await.ok()?;
                let remote_key = trust.get_public_key(&entry.node_id).await.ok()??;
                let msg = federated_entry_message(&entry);
                if remote_key.verify(&msg, &entry.signature).is_err() {
                    warn!("federation lookup: bad signature from {}", &peer_id[..16.min(peer_id.len())]);
                    return None;
                }
                Some(entry)
            }));
        }
        for handle in handles {
            if let Ok(Some(entry)) = handle.await { return Some(entry); }
        }
        None
    }

    /// Query TRUSTED, REACHABLE peers for semantically similar entries.
    /// Peers are sorted by average latency (fastest first).
    pub async fn lookup_semantic(
        &self,
        embedding:     &[f32],
        domain:        &str,
        sim_threshold: f64,
        limit:         usize,
    ) -> Option<(FederatedEntry, f64)> {
        if !self.enabled { return None; }

        let peers = self.reachable_peers_sorted().await;
        if peers.is_empty() { return None; }

        let req = SemanticLookupRequest {
            domain:        domain.to_string(),
            embedding:     embedding.to_vec(),
            sim_threshold,
            limit,
        };
        let req_body = serde_json::to_value(&req).ok()?;

        let mut handles = Vec::new();
        for peer in peers {
            let client  = self.client.clone();
            let trust   = self.trust.clone();
            let url     = format!("{}/v1/federation/semantic", peer.url);
            let peer_id = peer.id.clone();
            let body    = req_body.clone();
            handles.push(tokio::spawn(async move {
                let resp = client.post(&url).json(&body).send().await.ok()?;
                if !resp.status().is_success() { return None; }
                let hits: Vec<SemanticFederatedEntry> = resp.json().await.ok()?;
                let hit = hits.into_iter().next()?;
                let remote_key = trust.get_public_key(&hit.entry.node_id).await.ok()??;
                let msg = federated_entry_message(&hit.entry);
                if remote_key.verify(&msg, &hit.entry.signature).is_err() {
                    warn!("semantic federation: bad signature from {}", &peer_id[..16.min(peer_id.len())]);
                    return None;
                }
                Some((hit.entry, hit.similarity))
            }));
        }
        for handle in handles {
            if let Ok(Some(result)) = handle.await { return Some(result); }
        }
        None
    }

    /// Announce our shared hashes to all trusted peers.
    /// Piggybacks our trusted peer list for gossip-based discovery.
    pub async fn announce(&self, hashes: Vec<String>, our_url: &str) {
        if !self.enabled || hashes.is_empty() { return; }
        let peers = self.live_peers().await;
        if peers.is_empty() { return; }

        // Build gossip peer list: our trusted peers (excluding self).
        let gossip_peers: Vec<PeerDescriptor> = self.trust
            .list_trusted().await.unwrap_or_default()
            .into_iter()
            .filter(|r| !r.url.is_empty() && r.node_id != self.identity.fingerprint)
            .map(|r| PeerDescriptor {
                node_id:        r.node_id,
                url:            r.url,
                public_key_hex: r.public_key_hex,
            })
            .collect();

        let mut payload = AnnouncePayload::build(&self.identity, our_url, hashes);
        if !gossip_peers.is_empty() {
            payload.known_peers = Some(gossip_peers);
        }

        let body = match serde_json::to_value(&payload) {
            Ok(v)  => v,
            Err(e) => { warn!("announce serialize error: {e}"); return; }
        };
        for peer in peers {
            let url    = format!("{}/v1/federation/announce", peer.url);
            let client = self.client.clone();
            let b      = body.clone();
            tokio::spawn(async move { let _ = client.post(&url).json(&b).send().await; });
        }
    }

    /// Pull the trusted peer list from a single peer URL and register any
    /// unknown non-evicted peers so the mesh can be discovered transitively.
    /// Called at startup (against each config peer) and during hourly sync.
    pub async fn exchange_peers(&self, peer_url: &str) {
        if !self.enabled { return; }
        let url = format!("{}/v1/federation/peers/list", peer_url.trim_end_matches('/'));
        let peers: Vec<PeerDescriptor> = match self.client.get(&url).send().await {
            Ok(r) if r.status().is_success() => r.json().await.unwrap_or_default(),
            _ => return,
        };
        let own_id = &self.identity.fingerprint;
        let mut registered = 0usize;
        for p in peers {
            if p.node_id == *own_id || p.url.is_empty() { continue; }
            if self.trust.is_evicted(&p.node_id).await { continue; }
            if self.trust.is_trusted(&p.node_id).await { continue; } // already known
            match self.trust.register(&p.node_id, &p.public_key_hex, &p.url).await {
                Ok(_) => { registered += 1; }
                Err(e) => warn!("gossip register error for {}: {e}", &p.node_id[..16.min(p.node_id.len())]),
            }
        }
        if registered > 0 {
            info!("gossip: discovered {} new peer(s) from {}", registered, peer_url);
        }
    }

    // ── Revocation gossip ────────────────────────────────────────────────────

    /// Push a signed revocation to all trusted peers (fire-and-forget).
    /// Called immediately after a local eviction so the mesh learns fast.
    pub async fn push_revocation_to_peers(&self, rev: &RevocationRecord) {
        if !self.enabled { return; }
        let peers = self.live_peers().await;
        if peers.is_empty() { return; }
        let body = match serde_json::to_value(rev) {
            Ok(v)  => v,
            Err(e) => { warn!("revocation serialize error: {e}"); return; }
        };
        for peer in peers {
            let url    = format!("{}/v1/federation/revocations", peer.url);
            let client = self.client.clone();
            let b      = body.clone();
            tokio::spawn(async move { let _ = client.post(&url).json(&b).send().await; });
        }
    }

    /// Pull revocations from a single peer URL and apply any that are new and valid.
    /// Used both on startup sync and after receiving an announce.
    pub async fn pull_revocations_from_url(&self, peer_url: &str, cache: &CacheStore) -> usize {
        #[derive(serde::Deserialize)]
        struct RevocationListResponse { revocations: Vec<RevocationRecord> }

        let url = format!("{}/v1/federation/revocations", peer_url);
        let revocations: Vec<RevocationRecord> = match self.client.get(&url).send().await {
            Ok(r) if r.status().is_success() => {
                r.json::<RevocationListResponse>().await
                    .map(|w| w.revocations)
                    .unwrap_or_default()
            }
            _ => return 0,
        };
        let mut applied = 0usize;
        for rev in &revocations {
            match self.trust.apply_incoming_revocation(rev, cache).await {
                Ok(true) => {
                    info!("applied revocation for {} (from {})", &rev.node_id[..16.min(rev.node_id.len())], peer_url);
                    applied += 1;
                }
                Ok(false) => {}
                Err(e) => warn!("revocation apply error: {e}"),
            }
        }
        applied
    }

    /// Pull revocations from all trusted peers.  Called hourly and on startup.
    pub async fn sync_revocations(&self, cache: &CacheStore) -> usize {
        if !self.enabled { return 0; }
        let peers = self.live_peers().await;
        let mut total = 0usize;
        for peer in peers {
            total += self.pull_revocations_from_url(&peer.url, cache).await;
        }
        total
    }

    /// Return the live peers that are currently reachable, sorted by average
    /// health-check latency (fastest first).  Peers with no health data are
    /// included (benefit of the doubt) and placed after peers with known-good
    /// latency.
    async fn reachable_peers_sorted(&self) -> Vec<PeerNode> {
        let peers  = self.live_peers().await;
        let health = self.trust.list_peer_health().await.unwrap_or_default();

        // Build a latency map: node_id → avg_latency_ms (None = unreachable)
        let lat: std::collections::HashMap<&str, Option<f64>> = health
            .iter()
            .map(|h| (h.node_id.as_str(), if h.is_reachable { h.avg_latency_ms } else { None }))
            .collect();

        let mut filtered: Vec<PeerNode> = peers.into_iter()
            .filter(|p| {
                // Skip peers the health checker has marked unreachable.
                // Peers with no health record are included.
                lat.get(p.id.as_str()).map(|v| v.is_some()).unwrap_or(true)
            })
            .collect();

        // Fastest known peers first; unknown-latency peers at the end.
        filtered.sort_by(|a, b| {
            let la = lat.get(a.id.as_str()).and_then(|v| *v).unwrap_or(f64::MAX);
            let lb = lat.get(b.id.as_str()).and_then(|v| *v).unwrap_or(f64::MAX);
            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
        });

        filtered
    }

    /// Fetch distilled domain knowledge from all trusted reachable peers for `domain`.
    /// Returns one `PeerKnowledge` per peer that responded successfully.  Used by the
    /// Distiller to blend peer knowledge docs and contrast pairs into local synthesis.
    pub async fn fetch_peer_knowledge(&self, domain: &str) -> Vec<PeerKnowledge> {
        if !self.enabled { return vec![]; }
        let peers = self.reachable_peers_sorted().await;
        if peers.is_empty() { return vec![]; }
        let mut results = Vec::new();
        for peer in peers {
            let url = format!("{}/v1/federation/knowledge/{}", peer.url.trim_end_matches('/'), domain);
            match self.client.get(&url).send().await {
                Ok(r) if r.status().is_success() => {
                    match r.json::<PeerKnowledge>().await {
                        Ok(pk) => {
                            debug!("federation knowledge: fetched '{}' from {}", domain,
                                &peer.id[..16.min(peer.id.len())]);
                            results.push(pk);
                        }
                        Err(e) => debug!("federation knowledge: parse error from {}: {e}",
                            &peer.id[..16.min(peer.id.len())]),
                    }
                }
                _ => {}
            }
        }
        results
    }

    /// Count of currently trusted peers (excludes self, requires a non-empty URL).
    pub async fn peer_count(&self) -> usize {
        self.live_peers().await.len()
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a CacheEntry into the federation wire format, signed by this node's key.
pub fn entry_to_federated(entry: &CacheEntry, identity: &NodeIdentity) -> FederatedEntry {
    let mut fe = FederatedEntry {
        hash:       entry.id.clone(),
        response:   entry.response.clone(),
        model_used: entry.model_used.clone(),
        domain:     entry.domain.clone(),
        node_id:    identity.fingerprint.clone(),
        signature:  String::new(),
    };
    let msg      = federated_entry_message(&fe);
    fe.signature = identity.sign(&msg);
    fe
}

/// Canonical message to sign for a FederatedEntry.
/// Covers the full response via SHA256 so truncation cannot forge a valid entry.
pub fn federated_entry_message(fe: &FederatedEntry) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let resp_hash = hex::encode(Sha256::digest(fe.response.as_bytes()));
    format!("{}|{}|{}", fe.hash, fe.node_id, resp_hash).into_bytes()
}

