use anyhow::{bail, Result};
use serde::Deserialize;
use std::path::PathBuf;

/// Resolved API credentials — never stored in config.toml.
#[derive(Debug, Clone)]
pub struct Credentials {
    pub api_key: String,
}

/// Load credentials from (in priority order):
///   1. ANTHROPIC_API_KEY environment variable
///   2. ~/.claude/.credentials.json  (Claude desktop OAuth token)
pub fn load() -> Result<Credentials> {
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

#[derive(Deserialize)]
struct CredentialsFile {
    #[serde(rename = "claudeAiOauthToken")]
    claude_ai_oauth_token: Option<String>,
    api_key: Option<String>,
}

fn load_credentials_json(path: &PathBuf) -> Result<Credentials> {
    let text = std::fs::read_to_string(path)?;
    let file: CredentialsFile = serde_json::from_str(&text)?;

    let key = file.claude_ai_oauth_token
        .or(file.api_key)
        .filter(|s| !s.is_empty());

    match key {
        Some(k) => Ok(Credentials { api_key: k }),
        None => bail!("credentials.json has no usable key"),
    }
}
