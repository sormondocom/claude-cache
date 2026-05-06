use crate::config::AppConfig;
use crate::domain::{has_recency_signal, has_version_specifier, ShapeKey};

#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// How long to cache this response (seconds). None = do not cache.
    pub ttl_secs: Option<u64>,
    /// Whether to bypass the cache entirely for this request.
    pub bypass_cache: bool,
    /// Whether this entry may be shared with federation peers.
    pub shareable: bool,
}

impl CachePolicy {
    pub fn should_cache(&self) -> bool {
        !self.bypass_cache && self.ttl_secs.is_some()
    }
}

pub fn infer(shape: &ShapeKey, prompt: &str, cfg: &AppConfig) -> CachePolicy {
    let lower = prompt.to_lowercase();

    // Hard bypasses — recency or version-specific questions should not be cached long
    let has_recency = has_recency_signal(&lower);
    let has_version = has_version_specifier(prompt);

    if has_recency && !has_version {
        // Purely "what's new / latest" — too volatile to cache
        return CachePolicy {
            ttl_secs:    None,
            bypass_cache: true,
            shareable:   false,
        };
    }

    let base_ttl = cfg.domain_ttl(&shape.domain);

    let ttl = if has_version {
        // Version-specific: cache, but shorter — the answer may become stale when a new
        // release drops
        (base_ttl / 2).max(1800)
    } else {
        base_ttl
    };

    // Short code-gen questions can be shared; private/security context should not
    let shareable = is_shareable(&lower);

    CachePolicy {
        ttl_secs:    Some(ttl),
        bypass_cache: false,
        shareable,
    }
}

fn is_shareable(lower: &str) -> bool {
    // Heuristic: prompts containing personal/private indicators are not shareable
    const PRIVATE_INDICATORS: &[&str] = &[
        "my api key", "my secret", "my password", "my token", "my project",
        "my company", "internal", "confidential", "private", "proprietary",
        "our codebase", "our system", "production secret",
    ];
    !PRIVATE_INDICATORS.iter().any(|p| lower.contains(p))
}
