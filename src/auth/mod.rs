use anyhow::{bail, Result};
use arc_swap::ArcSwap;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;

/// Resolved API credentials — never stored in config.toml.
#[derive(Debug, Clone)]
pub struct Credentials {
    pub api_key: String,
}

impl Credentials {
    /// True when this is a real Anthropic API key (sk-ant-api…) suitable for
    /// direct calls to api.anthropic.com.
    pub fn is_api_key(&self) -> bool {
        self.api_key.starts_with("sk-ant-api")
    }

    /// True when this is a Claude.ai OAuth access token (sk-ant-oat…).
    /// Claude Max subscribers may have direct API access via Bearer auth;
    /// Claude Pro subscribers typically do not.  The proxy attempts the call
    /// either way and self-disables at runtime if access is denied.
    pub fn is_oauth_token(&self) -> bool {
        self.api_key.starts_with("sk-ant-oat")
    }
}

/// Thread-safe, hot-reloadable credential holder.
///
/// Wraps credentials in an `ArcSwap` so the background mtime-watcher and
/// the per-request 401-retry path can both refresh the token without a lock.
/// `Clone` is cheap — it shares the inner `Arc`.
#[derive(Clone)]
pub struct CredentialStore {
    inner: Arc<ArcSwap<Credentials>>,
}

impl CredentialStore {
    fn new(creds: Credentials) -> Self {
        Self { inner: Arc::new(ArcSwap::from_pointee(creds)) }
    }

    /// Return the current credentials.  Atomic load + refcount bump — very cheap.
    pub fn get(&self) -> Arc<Credentials> {
        self.inner.load_full()
    }

    /// Re-read credentials from the environment / credentials.json and swap them in.
    /// Called by the background watcher on mtime change and by the request path on 401.
    pub fn reload(&self) -> Result<()> {
        let fresh = load_from_env_or_file()?;
        self.inner.store(Arc::new(fresh));
        Ok(())
    }

    /// Create a store directly from a static key, bypassing env/file lookup.
    /// Intended for tests and programmatic configuration.
    pub fn from_key(api_key: impl Into<String>) -> Self {
        Self::new(Credentials { api_key: api_key.into() })
    }

    /// Path to `~/.claude/.credentials.json`, if it can be resolved.
    /// Exposed so the background watcher in main.rs can poll its mtime.
    pub fn credentials_path() -> Option<PathBuf> {
        credentials_json_path()
    }
}

/// Load credentials and return a hot-reloadable store.
pub fn load() -> Result<CredentialStore> {
    Ok(CredentialStore::new(load_from_env_or_file()?))
}

// ── Internal helpers ───────────────────────────────────────────────────────

fn load_from_env_or_file() -> Result<Credentials> {
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        if !key.is_empty() {
            return Ok(Credentials { api_key: key });
        }
    }

    if let Some(path) = credentials_json_path() {
        if path.exists() {
            if let Ok(creds) = load_credentials_json(&path) {
                return Ok(creds);
            }
        }
    }

    bail!(
        "No API credentials found. \
         Set ANTHROPIC_API_KEY or sign in with the Claude desktop app."
    )
}

fn credentials_json_path() -> Option<PathBuf> {
    let home = if cfg!(windows) {
        std::env::var("USERPROFILE").ok()?
    } else {
        std::env::var("HOME").ok()?
    };
    Some(PathBuf::from(home).join(".claude").join(".credentials.json"))
}

/// Claude Desktop stores OAuth creds as a nested object:
/// { "claudeAiOauth": { "accessToken": "sk-ant-oat01-...", ... } }
/// Older Claude Code CLI versions used a flat string field instead.
#[derive(Deserialize)]
struct OAuthObject {
    #[serde(rename = "accessToken")]
    access_token: Option<String>,
}

#[derive(Deserialize)]
struct CredentialsFile {
    /// Claude Desktop nested format
    #[serde(rename = "claudeAiOauth")]
    claude_ai_oauth: Option<OAuthObject>,
    /// Legacy flat format from older Claude Code CLI
    #[serde(rename = "claudeAiOauthToken")]
    claude_ai_oauth_token: Option<String>,
    api_key: Option<String>,
}

fn load_credentials_json(path: &PathBuf) -> Result<Credentials> {
    let text = std::fs::read_to_string(path)?;
    let file: CredentialsFile = serde_json::from_str(&text)?;

    let key = file.claude_ai_oauth
            .and_then(|o| o.access_token)
        .or(file.claude_ai_oauth_token)
        .or(file.api_key)
        .filter(|s| !s.is_empty());

    match key {
        Some(k) => Ok(Credentials { api_key: k }),
        None    => bail!("credentials.json has no usable key"),
    }
}
