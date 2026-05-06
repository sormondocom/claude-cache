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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host:    String,
    pub port:    u16,
    #[serde(default)]
    pub node_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiConfig {
    pub model:    String,
    pub base_url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalConfig {
    pub enabled:         bool,
    pub backend:         String,
    pub base_url:        String,
    pub model_id:        String,
    pub confidence_floor: f64,
    pub timeout_secs:    u64,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FederationConfig {
    pub enabled:           bool,
    pub share_cache:       bool,
    pub peers:             Vec<String>,
    pub lookup_timeout_ms: u64,
}

impl AppConfig {
    pub fn load(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config from {path}"))?;
        let mut cfg: AppConfig = toml::from_str(&text)
            .with_context(|| "parsing config.toml")?;

        if cfg.server.node_id.is_empty() {
            cfg.server.node_id = uuid::Uuid::new_v4().to_string();
            cfg.persist_node_id(path)?;
        }
        Ok(cfg)
    }

    /// Write the generated node_id back into config.toml so it persists across restarts.
    fn persist_node_id(&self, path: &str) -> Result<()> {
        let text = std::fs::read_to_string(path)?;
        let new_text = if text.contains("node_id = \"\"") {
            text.replace(
                "node_id = \"\"",
                &format!("node_id = \"{}\"", self.server.node_id),
            )
        } else {
            // Append under [server] — simple best-effort
            text
        };
        std::fs::write(path, new_text)?;
        Ok(())
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
                host:    "127.0.0.1".into(),
                port:    3000,
                node_id: String::new(),
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
        }
    }
}
