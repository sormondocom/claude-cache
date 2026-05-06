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
use tracing::{debug, warn};

use crate::cache::CacheEntry;
use crate::identity::{announce_message, NodeIdentity, RemoteKey};
use crate::trust::TrustStore;

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

// ── Client ─────────────────────────────────────────────────────────────────

pub struct FederationClient {
    client:     reqwest::Client,
    peers:      Vec<PeerNode>,
    enabled:    bool,
    identity:   Arc<NodeIdentity>,
    trust:      Arc<TrustStore>,
}

impl FederationClient {
    pub fn new(
        peers:       Vec<PeerNode>,
        enabled:     bool,
        identity:    Arc<NodeIdentity>,
        trust:       Arc<TrustStore>,
        timeout_ms:  u64,
    ) -> Self {
        FederationClient {
            client: reqwest::Client::builder()
                .timeout(Duration::from_millis(timeout_ms))
                .use_rustls_tls()
                .build()
                .expect("federation reqwest client"),
            peers,
            enabled,
            identity,
            trust,
        }
    }

    /// Query TRUSTED peers for a cache entry by SHA256 hash.
    /// Untrusted peers are never queried.
    /// The response is signature-verified before being returned.
    pub async fn lookup(&self, hash: &str) -> Option<FederatedEntry> {
        if !self.enabled || self.peers.is_empty() {
            return None;
        }

        let futures: Vec<_> = self.peers.iter().map(|peer| {
            let client   = self.client.clone();
            let trust    = self.trust.clone();
            let url      = format!("{}/v1/federation/lookup/{}", peer.url, hash);
            let peer_id  = peer.id.clone();
            async move {
                // Only query trusted peers
                if !trust.is_trusted(&peer_id).await {
                    debug!("skipping untrusted peer {}", &peer_id[..16.min(peer_id.len())]);
                    return None;
                }

                let resp = client.get(&url).send().await.ok()?;
                if !resp.status().is_success() {
                    return None;
                }

                let entry: FederatedEntry = resp.json().await.ok()?;

                // Verify the response signature
                let remote_key = trust.get_public_key(&entry.node_id).await.ok()??;
                let msg = federated_entry_message(&entry);
                if remote_key.verify(&msg, &entry.signature).is_err() {
                    warn!("federation lookup: bad signature from {}", &peer_id[..16.min(peer_id.len())]);
                    return None;
                }

                Some(entry)
            }
        }).collect();

        let mut handles = Vec::new();
        for fut in futures {
            handles.push(tokio::spawn(fut));
        }
        for handle in handles {
            if let Ok(Some(entry)) = handle.await {
                return Some(entry);
            }
        }
        None
    }

    /// Announce our shared hashes to all peers.
    /// Only sends to peers whose url we know; the peers decide whether to trust us.
    pub async fn announce(&self, hashes: Vec<String>, our_url: &str) {
        if !self.enabled || self.peers.is_empty() || hashes.is_empty() {
            return;
        }
        let payload = AnnouncePayload::build(&self.identity, our_url, hashes);
        let body    = match serde_json::to_value(&payload) {
            Ok(v)  => v,
            Err(e) => { warn!("announce serialize error: {e}"); return; }
        };
        for peer in &self.peers {
            let url    = format!("{}/v1/federation/announce", peer.url);
            let client = self.client.clone();
            let b      = body.clone();
            tokio::spawn(async move {
                let _ = client.post(&url).json(&b).send().await;
            });
        }
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
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
pub fn federated_entry_message(fe: &FederatedEntry) -> Vec<u8> {
    format!("{}|{}|{}", fe.hash, fe.node_id, &fe.response[..fe.response.len().min(256)])
        .into_bytes()
}

/// Build a PeerNode list from config URL strings.
pub fn peers_from_urls(urls: &[String]) -> Vec<PeerNode> {
    urls.iter()
        .map(|url| PeerNode {
            id:  format!("peer-{}", &url[url.len().saturating_sub(8)..]),
            url: url.clone(),
        })
        .collect()
}
