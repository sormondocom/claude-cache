use std::{collections::HashMap, path::PathBuf};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    pub server:     ServerConfig,
    pub api:        ApiConfig,
    pub local:      LocalConfig,
    pub embedding:  EmbeddingConfig,
    pub cache:      CacheConfig,
    pub routing:    RoutingConfig,
    pub budget:     BudgetConfig,
    pub federation: FederationConfig,
    #[serde(default)]
    pub node:       NodeConfig,
    #[serde(default)]
    pub health:     HealthConfig,
}

// ── Server ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

// ── API / Local / Embedding / Cache / Routing / Budget ──────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiConfig {
    pub model:    String,
    pub base_url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalConfig {
    pub enabled:          bool,
    pub backend:          String,
    pub base_url:         String,
    pub model_id:         String,
    pub confidence_floor: f64,
    pub timeout_secs:     u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub enabled:       bool,
    pub base_url:      String,
    pub model:         String,
    pub sim_threshold: f64,
    pub dimensions:    usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CacheConfig {
    pub db_path:          String,
    pub max_size_mb:      u64,
    pub default_ttl_secs: u64,
    #[serde(default)]
    pub domain_ttl:       HashMap<String, u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RoutingConfig {
    pub novelty_threshold:     f64,
    pub complexity_threshold:  f64,
    pub consequence_threshold: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BudgetConfig {
    pub db_path:           String,
    pub daily_limit_usd:   f64,
    pub warn_at_pct:       u8,
    pub input_per_1k_usd:  f64,
    pub output_per_1k_usd: f64,
}

// ── Federation ───────────────────────────────────────────────────────────────

/// A statically-configured federation peer.  The `node_id` (Ed25519 fingerprint)
/// is required and obtained by running `claude-cache identity` on the peer.
/// `public_key_hex` is optional but strongly recommended — without it, federated
/// responses from this peer cannot be signature-verified until after their first
/// announce reaches us.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PeerConfig {
    pub url:     String,
    pub node_id: String,
    #[serde(default)]
    pub public_key_hex: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FederationConfig {
    pub enabled:           bool,
    pub share_cache:       bool,
    #[serde(default)]
    pub peers:             Vec<PeerConfig>,
    pub lookup_timeout_ms: u64,
}

// ── Health checks ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HealthConfig {
    /// Enable background peer health checks.
    pub enabled:           bool,
    /// How often to check each peer (seconds).
    pub interval_secs:     u64,
    /// HTTP timeout per health check request (milliseconds).
    pub timeout_ms:        u64,
    /// Number of consecutive failures before a peer is marked unreachable.
    pub failure_threshold: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        HealthConfig {
            enabled:           true,
            interval_secs:     60,
            timeout_ms:        2000,
            failure_threshold: 3,
        }
    }
}

// ── Node role ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum NodeRole {
    /// Command-and-control node: head node, can counter-sign peers, optional
    /// auto-promote.  Typically runs with higher budget and always reaches the
    /// Anthropic API directly (no local model routing for CNC responses).
    Cnc,
    /// Standard client node.  Participates in the mesh, routes through the
    /// local cascade, can bootstrap trust from a CNC.
    Client,
}

impl Default for NodeRole {
    fn default() -> Self { NodeRole::Client }
}

/// Node-level identity and bootstrapping config.  All fields are optional and
/// default to "plain client" behaviour when omitted.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NodeConfig {
    /// Role of this node in the federation mesh.
    #[serde(default)]
    pub role: NodeRole,

    /// CNC only: automatically promote every announcing peer to trusted.
    /// Keep false in production; use manual `POST /v1/trust/:node_id` instead.
    #[serde(default)]
    pub auto_promote_peers: bool,

    /// Client only: URL of the CNC node to announce to at startup.
    #[serde(default)]
    pub cnc_url: String,

    /// Client only: Ed25519 fingerprint of the CNC node.  Required when
    /// `cnc_url` is set so we can auto-trust the CNC on first contact.
    #[serde(default)]
    pub cnc_node_id: String,
}

impl Default for NodeConfig {
    fn default() -> Self {
        NodeConfig {
            role:               NodeRole::Client,
            auto_promote_peers: false,
            cnc_url:            String::new(),
            cnc_node_id:        String::new(),
        }
    }
}

// ── AppConfig ─────────────────────────────────────────────────────────────────

impl AppConfig {
    pub fn load(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config from {path}"))?;
        toml::from_str(&text)
            .with_context(|| "parsing config.toml")
    }

    pub fn domain_ttl(&self, domain: &str) -> u64 {
        self.cache
            .domain_ttl
            .get(domain)
            .copied()
            .unwrap_or(self.cache.default_ttl_secs)
    }

    pub fn db_path(&self) -> PathBuf {
        PathBuf::from(&self.cache.db_path)
    }

    pub fn budget_db_path(&self) -> PathBuf {
        PathBuf::from(&self.budget.db_path)
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 3000,
            },
            api: ApiConfig {
                model:    "claude-sonnet-4-6".into(),
                base_url: "https://api.anthropic.com".into(),
            },
            local: LocalConfig {
                enabled:          true,
                backend:          "ollama".into(),
                base_url:         "http://localhost:11434".into(),
                model_id:         "gemma4".into(),
                confidence_floor: 0.75,
                timeout_secs:     120,
            },
            embedding: EmbeddingConfig {
                enabled:       true,
                base_url:      "http://localhost:11434".into(),
                model:         "nomic-embed-text".into(),
                sim_threshold: 0.88,
                dimensions:    768,
            },
            cache: CacheConfig {
                db_path:          "claude-cache.db".into(),
                max_size_mb:      500,
                default_ttl_secs: 3600,
                domain_ttl:       HashMap::new(),
            },
            routing: RoutingConfig {
                novelty_threshold:     0.60,
                complexity_threshold:  0.40,
                consequence_threshold: 0.30,
            },
            budget: BudgetConfig {
                db_path:           "claude-cache.budget.db".into(),
                daily_limit_usd:   0.50,
                warn_at_pct:       80,
                input_per_1k_usd:  0.003,
                output_per_1k_usd: 0.015,
            },
            federation: FederationConfig {
                enabled:           false,
                share_cache:       false,
                peers:             Vec::new(),
                lookup_timeout_ms: 500,
            },
            node:   NodeConfig::default(),
            health: HealthConfig::default(),
        }
    }
}
