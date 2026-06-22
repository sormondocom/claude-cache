/// Node identity — Ed25519 keypair with PGP-style web-of-trust semantics.
///
/// Every node has a stable signing keypair generated on first run and persisted
/// to `node_identity.key` alongside the databases.  The public key fingerprint
/// (SHA256 of the raw public key bytes, hex-encoded) is used as the node's
/// canonical identity string throughout the system.
///
/// Announcements to federation peers are signed with this key.  Peers verify
/// the signature and check the signer's trust state before accepting any data.

use anyhow::{Context, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};
use std::path::Path;

// ── Key material ────────────────────────────────────────────────────────────

/// Full keypair held by this node.
#[derive(Clone)]
pub struct NodeIdentity {
    signing_key:   SigningKey,
    verifying_key: VerifyingKey,
    pub fingerprint: String,
    pub public_key_hex: String,
}

impl NodeIdentity {
    /// Generate a fresh keypair.  Call `save()` immediately after to persist.
    pub fn generate() -> Self {
        let signing_key   = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let fingerprint   = fingerprint_of(verifying_key.as_bytes());
        let public_key_hex = hex::encode(verifying_key.as_bytes());
        NodeIdentity { signing_key, verifying_key, fingerprint, public_key_hex }
    }

    /// Load from a file produced by `save()`, or generate + save if the file
    /// does not exist.
    pub fn load_or_generate(path: &str) -> Result<Self> {
        if Path::new(path).exists() {
            Self::load(path)
        } else {
            let id = Self::generate();
            id.save(path)?;
            Ok(id)
        }
    }

    fn load(path: &str) -> Result<Self> {
        let hex = std::fs::read_to_string(path)
            .with_context(|| format!("reading identity file {path}"))?;
        let hex = hex.trim();
        let bytes: [u8; 32] = hex::decode(hex)
            .context("decoding identity hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("identity key must be 32 bytes"))?;
        let signing_key   = SigningKey::from_bytes(&bytes);
        let verifying_key = signing_key.verifying_key();
        let fingerprint   = fingerprint_of(verifying_key.as_bytes());
        let public_key_hex = hex::encode(verifying_key.as_bytes());
        Ok(NodeIdentity { signing_key, verifying_key, fingerprint, public_key_hex })
    }

    /// Persist the private key bytes (hex) to `path`.  Mode 0600 on Unix.
    pub fn save(&self, path: &str) -> Result<()> {
        let hex = hex::encode(self.signing_key.to_bytes());
        std::fs::write(path, &hex)
            .with_context(|| format!("writing identity to {path}"))?;

        // Restrict permissions so the private key isn't world-readable.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        }
        #[cfg(windows)]
        {
            // Remove inherited ACEs, grant only the current user full control.
            if let Ok(username) = std::env::var("USERNAME") {
                let _ = std::process::Command::new("icacls")
                    .args([path, "/inheritance:r", "/grant:r", &format!("{username}:(F)")])
                    .output();
            }
        }

        Ok(())
    }

    /// Sign an arbitrary message.  Returns the 64-byte signature as hex.
    pub fn sign(&self, msg: &[u8]) -> String {
        let sig: Signature = self.signing_key.sign(msg);
        hex::encode(sig.to_bytes())
    }

    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }
}

// ── Public-key-only handle (for remote nodes) ──────────────────────────────

#[derive(Debug, Clone)]
pub struct RemoteKey {
    pub verifying_key:  VerifyingKey,
    pub fingerprint:    String,
    pub public_key_hex: String,
}

impl RemoteKey {
    pub fn from_hex(public_key_hex: &str) -> Result<Self> {
        let bytes: [u8; 32] = hex::decode(public_key_hex)
            .context("decoding remote public key")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("remote public key must be 32 bytes"))?;
        let verifying_key  = VerifyingKey::from_bytes(&bytes)
            .context("invalid Ed25519 public key")?;
        let fingerprint    = fingerprint_of(&bytes);
        Ok(RemoteKey {
            verifying_key,
            fingerprint,
            public_key_hex: public_key_hex.to_string(),
        })
    }

    /// Verify a hex-encoded signature over `msg`.
    pub fn verify(&self, msg: &[u8], sig_hex: &str) -> Result<()> {
        let sig_bytes: [u8; 64] = hex::decode(sig_hex)
            .context("decoding signature hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("signature must be 64 bytes"))?;
        let sig = Signature::from_bytes(&sig_bytes);
        self.verifying_key
            .verify(msg, &sig)
            .context("signature verification failed")
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Canonical fingerprint: SHA256(pubkey_bytes) as lowercase hex.
/// First 16 chars used as short display ID.
pub fn fingerprint_of(pubkey_bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(pubkey_bytes);
    hex::encode(h.finalize())
}

/// Canonical message to sign when revoking a node.
/// Format is deterministic so any verifier can reconstruct it.
pub fn revocation_message(node_id: &str, reason: &str) -> Vec<u8> {
    format!("revoke|{}|{}", node_id, reason).into_bytes()
}

/// Canonical message to sign for an announce payload.
/// Format: `node_id|url|public_key_hex\n<newline-joined sorted hashes>`
/// Using a deterministic text format avoids JSON serialization ambiguity.
pub fn announce_message(node_id: &str, url: &str, public_key_hex: &str, hashes: &[String]) -> Vec<u8> {
    let mut sorted = hashes.to_vec();
    sorted.sort_unstable();
    let hash_list = sorted.join("\n");
    format!("{node_id}|{url}|{public_key_hex}\n{hash_list}").into_bytes()
}
