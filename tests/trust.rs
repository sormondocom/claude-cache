//! Integration tests for the identity, trust, and revocation gossip systems.

use std::sync::Arc;
use tempfile::tempdir;

use claude_cache::{
    cache::CacheStore,
    identity::{NodeIdentity, announce_message, revocation_message},
    trust::{NodeTrustState, RevocationRecord, TrustStore},
};

// ── Helpers ────────────────────────────────────────────────────────────────────

async fn make_trust_store() -> (Arc<TrustStore>, Arc<CacheStore>, Arc<NodeIdentity>, tempfile::TempDir) {
    let dir        = tempdir().unwrap();
    let trust_path = dir.path().join("trust.db").to_str().unwrap().to_string();
    let cache_path = dir.path().join("cache.db").to_str().unwrap().to_string();
    let identity   = Arc::new(NodeIdentity::generate());

    let trust = Arc::new(TrustStore::open(&trust_path, &identity.fingerprint).await.unwrap());
    let cache = Arc::new(CacheStore::open(&cache_path, &identity.fingerprint).await.unwrap());

    (trust, cache, identity, dir)
}

fn make_identity() -> Arc<NodeIdentity> {
    Arc::new(NodeIdentity::generate())
}

// ── Identity tests ─────────────────────────────────────────────────────────────

#[test]
fn node_id_is_stable_for_same_key() {
    // Fingerprint must be deterministic: same private key → same fingerprint
    let id1 = NodeIdentity::generate();
    let fp1 = id1.fingerprint.clone();
    // Encode/decode round-trip via hex (no file I/O needed for determinism check)
    let pubkey_hex = id1.public_key_hex.clone();
    let remote = claude_cache::identity::RemoteKey::from_hex(&pubkey_hex).unwrap();
    assert_eq!(remote.fingerprint, fp1, "fingerprint must match after round-trip");
}

#[test]
fn different_identities_have_different_fingerprints() {
    let a = NodeIdentity::generate();
    let b = NodeIdentity::generate();
    assert_ne!(a.fingerprint, b.fingerprint);
    assert_ne!(a.public_key_hex, b.public_key_hex);
}

#[test]
fn announce_signature_verifies() {
    let id      = NodeIdentity::generate();
    let url     = "http://127.0.0.1:3000";
    let hashes  = vec!["abc123".to_string(), "def456".to_string()];
    let payload = claude_cache::federation::AnnouncePayload::build(&id, url, hashes);

    assert!(payload.verify_self().is_ok(), "self-signature must verify");
}

#[test]
fn announce_tampered_node_id_fails_verify() {
    let id      = NodeIdentity::generate();
    let other   = NodeIdentity::generate();
    let url     = "http://127.0.0.1:3000";
    let mut payload = claude_cache::federation::AnnouncePayload::build(&id, url, vec![]);

    // Swap in a different node_id
    payload.node_id = other.fingerprint.clone();

    assert!(payload.verify_self().is_err(), "tampered node_id must fail verification");
}

#[test]
fn revocation_signature_verifies() {
    let id  = NodeIdentity::generate();
    let msg = revocation_message("some-node-id", "compromised");
    let sig = id.sign(&msg);

    let remote = claude_cache::identity::RemoteKey::from_hex(&id.public_key_hex).unwrap();
    assert!(remote.verify(&msg, &sig).is_ok());
}

#[test]
fn revocation_wrong_key_fails() {
    let signer  = NodeIdentity::generate();
    let impostor = NodeIdentity::generate();

    let msg = revocation_message("target", "bad actor");
    let sig = signer.sign(&msg);

    let impostor_key = claude_cache::identity::RemoteKey::from_hex(&impostor.public_key_hex).unwrap();
    assert!(impostor_key.verify(&msg, &sig).is_err());
}

// ── Trust store tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn new_node_starts_untrusted() {
    let (trust, _, _, _dir) = make_trust_store().await;
    let peer = make_identity();

    let state = trust
        .register(&peer.fingerprint, &peer.public_key_hex, "http://peer:3000")
        .await
        .unwrap();

    assert_eq!(state, NodeTrustState::Untrusted);
    assert!(!trust.is_trusted(&peer.fingerprint).await);
}

#[tokio::test]
async fn promote_changes_state_to_trusted() {
    let (trust, _, our_id, _dir) = make_trust_store().await;
    let peer = make_identity();

    trust.register(&peer.fingerprint, &peer.public_key_hex, "http://peer:3000").await.unwrap();
    trust.promote(&peer.fingerprint, &our_id.fingerprint, false).await.unwrap();

    assert!(trust.is_trusted(&peer.fingerprint).await);
    let state = trust.get_state(&peer.fingerprint).await.unwrap();
    assert!(matches!(state, NodeTrustState::Trusted { .. }));
}

#[tokio::test]
async fn head_node_promotion_marks_is_head() {
    let (trust, _, our_id, _dir) = make_trust_store().await;
    let peer = make_identity();

    trust.register(&peer.fingerprint, &peer.public_key_hex, "http://peer:3000").await.unwrap();
    trust.promote(&peer.fingerprint, &our_id.fingerprint, true).await.unwrap(); // is_head=true

    let record = trust.get_record(&peer.fingerprint).await.unwrap().unwrap();
    assert!(record.is_head);
}

#[tokio::test]
async fn evict_changes_state_to_evicted() {
    let (trust, cache, our_id, _dir) = make_trust_store().await;
    let peer = make_identity();

    trust.register(&peer.fingerprint, &peer.public_key_hex, "http://peer:3000").await.unwrap();
    trust.promote(&peer.fingerprint, &our_id.fingerprint, false).await.unwrap();

    let msg = revocation_message(&peer.fingerprint, "test eviction");
    let sig = our_id.sign(&msg);

    trust.evict(&peer.fingerprint, "test eviction", &our_id.fingerprint, &sig, &cache)
        .await
        .unwrap();

    assert!(trust.is_evicted(&peer.fingerprint).await);
    assert!(!trust.is_trusted(&peer.fingerprint).await);
}

#[tokio::test]
async fn evict_purges_cache_entries_from_that_node() {
    let (trust, cache, our_id, _dir) = make_trust_store().await;
    let peer = make_identity();

    // Register and trust peer so it can write cache entries
    trust.register(&peer.fingerprint, &peer.public_key_hex, "http://peer:3000").await.unwrap();
    trust.promote(&peer.fingerprint, &our_id.fingerprint, false).await.unwrap();

    // Seed two cache entries attributed to the peer node
    let shape = claude_cache::domain::ShapeKey {
        domain: "rust".into(), intent: "explain".into(), complexity: 0.3
    };
    // We need a CacheStore opened as the peer's node to write entries attributed to peer
    let dir2       = tempdir().unwrap();
    let cache2_path = dir2.path().join("c2.db").to_str().unwrap().to_string();
    let peer_cache = CacheStore::open(&cache2_path, &peer.fingerprint).await.unwrap();

    peer_cache.store(&shape, "rust question 1", r#"{"id":"r1","type":"message","role":"assistant","content":[{"type":"text","text":"ans1"}],"model":"x","usage":{"input_tokens":1,"output_tokens":1}}"#,
        "anthropic", None, Some(3600), true).await.unwrap();
    peer_cache.store(&shape, "rust question 2", r#"{"id":"r2","type":"message","role":"assistant","content":[{"type":"text","text":"ans2"}],"model":"x","usage":{"input_tokens":1,"output_tokens":1}}"#,
        "anthropic", None, Some(3600), true).await.unwrap();

    let stats_before = peer_cache.stats().await.unwrap();
    assert_eq!(stats_before.total_entries, 2);

    // Evict the peer via our trust store, purging entries from peer_cache
    let msg = revocation_message(&peer.fingerprint, "compromised");
    let sig = our_id.sign(&msg);
    trust.evict(&peer.fingerprint, "compromised", &our_id.fingerprint, &sig, &peer_cache)
        .await
        .unwrap();

    let stats_after = peer_cache.stats().await.unwrap();
    assert_eq!(stats_after.total_entries, 0, "all peer entries must be purged on eviction");
}

// ── Revocation gossip tests ────────────────────────────────────────────────────

#[tokio::test]
async fn apply_incoming_revocation_from_trusted_revoker() {
    let (trust, cache, our_id, _dir) = make_trust_store().await;
    let revoker = make_identity();
    let target  = make_identity();

    // Register and trust the revoker
    trust.register(&revoker.fingerprint, &revoker.public_key_hex, "http://revoker:3000").await.unwrap();
    trust.promote(&revoker.fingerprint, &our_id.fingerprint, false).await.unwrap();

    // Register the target (but don't need to trust it for this test)
    trust.register(&target.fingerprint, &target.public_key_hex, "http://target:3000").await.unwrap();

    // Revoker signs the revocation
    let msg = revocation_message(&target.fingerprint, "misbehaving");
    let sig = revoker.sign(&msg);

    let rev = RevocationRecord {
        node_id:    target.fingerprint.clone(),
        revoked_by: revoker.fingerprint.clone(),
        reason:     "misbehaving".into(),
        signature:  sig,
        revoked_at: 0,
    };

    let applied = trust.apply_incoming_revocation(&rev, &cache).await.unwrap();
    assert!(applied, "revocation from trusted revoker must be applied");
    assert!(trust.is_evicted(&target.fingerprint).await);
}

#[tokio::test]
async fn apply_incoming_revocation_from_untrusted_revoker_is_rejected() {
    let (trust, cache, _our_id, _dir) = make_trust_store().await;
    let revoker = make_identity(); // NOT registered, NOT trusted
    let target  = make_identity();

    trust.register(&target.fingerprint, &target.public_key_hex, "http://target:3000").await.unwrap();

    let msg = revocation_message(&target.fingerprint, "test");
    let sig = revoker.sign(&msg);

    let rev = RevocationRecord {
        node_id:    target.fingerprint.clone(),
        revoked_by: revoker.fingerprint.clone(),
        reason:     "test".into(),
        signature:  sig,
        revoked_at: 0,
    };

    let applied = trust.apply_incoming_revocation(&rev, &cache).await.unwrap();
    assert!(!applied, "revocation from untrusted revoker must be rejected");
    assert!(!trust.is_evicted(&target.fingerprint).await, "target must not be evicted");
}

#[tokio::test]
async fn apply_incoming_revocation_with_bad_signature_is_rejected() {
    let (trust, cache, our_id, _dir) = make_trust_store().await;
    let revoker = make_identity();
    let target  = make_identity();

    // Trust the revoker
    trust.register(&revoker.fingerprint, &revoker.public_key_hex, "http://revoker:3000").await.unwrap();
    trust.promote(&revoker.fingerprint, &our_id.fingerprint, false).await.unwrap();
    trust.register(&target.fingerprint, &target.public_key_hex, "http://target:3000").await.unwrap();

    // Sign with WRONG key
    let impostor = make_identity();
    let msg = revocation_message(&target.fingerprint, "forged");
    let sig = impostor.sign(&msg); // impostor signs, not revoker

    let rev = RevocationRecord {
        node_id:    target.fingerprint.clone(),
        revoked_by: revoker.fingerprint.clone(), // claims to be revoker
        reason:     "forged".into(),
        signature:  sig,
        revoked_at: 0,
    };

    let applied = trust.apply_incoming_revocation(&rev, &cache).await.unwrap();
    assert!(!applied, "forged signature must be rejected");
    assert!(!trust.is_evicted(&target.fingerprint).await);
}

#[tokio::test]
async fn applying_same_revocation_twice_is_idempotent() {
    let (trust, cache, our_id, _dir) = make_trust_store().await;
    let revoker = make_identity();
    let target  = make_identity();

    trust.register(&revoker.fingerprint, &revoker.public_key_hex, "http://r:3000").await.unwrap();
    trust.promote(&revoker.fingerprint, &our_id.fingerprint, false).await.unwrap();
    trust.register(&target.fingerprint, &target.public_key_hex, "http://t:3000").await.unwrap();

    let msg = revocation_message(&target.fingerprint, "once");
    let sig = revoker.sign(&msg);
    let rev = RevocationRecord {
        node_id: target.fingerprint.clone(), revoked_by: revoker.fingerprint.clone(),
        reason: "once".into(), signature: sig.clone(), revoked_at: 0,
    };

    let first  = trust.apply_incoming_revocation(&rev, &cache).await.unwrap();
    let second = trust.apply_incoming_revocation(&rev, &cache).await.unwrap();

    assert!(first,   "first application must succeed");
    assert!(!second, "second application must return false (already evicted)");
    assert!(trust.is_evicted(&target.fingerprint).await);
}

#[tokio::test]
async fn revocations_are_listed_for_gossip() {
    let (trust, cache, our_id, _dir) = make_trust_store().await;
    let target = make_identity();

    trust.register(&target.fingerprint, &target.public_key_hex, "http://t:3000").await.unwrap();

    let msg = revocation_message(&target.fingerprint, "listed");
    let sig = our_id.sign(&msg);

    trust.evict(&target.fingerprint, "listed", &our_id.fingerprint, &sig, &cache)
        .await
        .unwrap();

    let revocations = trust.list_revocations().await.unwrap();
    assert!(!revocations.is_empty());
    assert_eq!(revocations[0].node_id, target.fingerprint);
}

// ── Head-node auto-promotion test ─────────────────────────────────────────────

#[tokio::test]
async fn head_node_counter_signature_auto_promotes_peer() {
    let (trust, _, our_id, _dir) = make_trust_store().await;
    let head = make_identity();
    let peer = make_identity();

    // Establish head node as trusted head
    trust.register(&head.fingerprint, &head.public_key_hex, "http://head:3000").await.unwrap();
    trust.promote(&head.fingerprint, &our_id.fingerprint, true).await.unwrap(); // is_head=true

    // Register peer (starts untrusted)
    trust.register(&peer.fingerprint, &peer.public_key_hex, "http://peer:3000").await.unwrap();
    assert!(!trust.is_trusted(&peer.fingerprint).await);

    // Head signs a counter-signature for peer
    let msg_for_peer = announce_message(&peer.fingerprint, "http://peer:3000", &peer.public_key_hex, &[]);
    let counter_sig  = head.sign(&msg_for_peer);

    // Build an announce with counter-signature
    let mut payload = claude_cache::federation::AnnouncePayload::build(&peer, "http://peer:3000", vec![]);
    payload.countersigned_by  = Some(head.fingerprint.clone());
    payload.counter_signature = Some(counter_sig);

    // Simulate the server handler logic: verify counter, auto-promote
    let head_key = trust.get_public_key(&head.fingerprint).await.unwrap().unwrap();
    if payload.verify_counter(&head_key).is_ok() {
        trust.auto_promote_if_head_signed(&peer.fingerprint, &head.fingerprint).await.unwrap();
    }

    assert!(trust.is_trusted(&peer.fingerprint).await, "peer auto-promoted by head node");
}
