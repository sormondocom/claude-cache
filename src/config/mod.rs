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
    #[serde(default)]
    pub limits:     LimitsConfig,
    #[serde(default)]
    pub learning:   LearningConfig,
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
    /// Set to false to disable all upstream Anthropic API calls.
    /// The proxy will serve from cache and local model only.
    #[serde(default = "default_true")]
    pub enabled:  bool,
    pub model:    String,
    pub base_url: String,
    /// Which upstream API backend to use for cache misses.
    /// "anthropic" — direct HTTPS to api.anthropic.com (requires API key or
    ///   a Claude Max subscription with API access).
    /// "claude_code" — spawns the local `claude --print` CLI subprocess; works
    ///   with any Pro/Max subscription as long as `claude` is in PATH.
    ///
    /// Auto-detected: if ANTHROPIC_API_KEY is not set in the environment, the
    /// proxy automatically uses "claude_code" regardless of this field.
    #[serde(default = "default_api_backend")]
    pub backend: String,
    /// How many times to retry a request that fails with an overloaded_error
    /// (HTTP 429 capacity pressure, not a rate limit). Each retry uses
    /// exponential backoff starting at retry_delay_ms.  Set to 0 to disable.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Base delay in milliseconds between retries. Doubles on each attempt
    /// (capped at 8× base): 500 → 1000 → 2000 → 2000 …
    #[serde(default = "default_retry_delay_ms")]
    pub retry_delay_ms: u64,
    /// HTTP request timeout in seconds for upstream Anthropic calls or the
    /// claude CLI subprocess.  Increase for large max_tokens or extended
    /// thinking.  Default is 300s (5 minutes).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Maximum number of concurrent `claude` CLI subprocesses.
    /// Only applies when backend = "claude_code" (or auto-selected).
    /// 0 = unlimited (not recommended — the OS will cap you less gracefully).
    #[serde(default = "default_claude_code_max_concurrency")]
    pub claude_code_max_concurrency: usize,
    /// Seconds a request waits for a free process slot before failing with a
    /// capacity error.  Only relevant when all claude_code_max_concurrency
    /// slots are occupied.
    #[serde(default = "default_claude_code_queue_timeout_secs")]
    pub claude_code_queue_timeout_secs: u64,
}

fn default_api_backend()                    -> String { "anthropic".to_string() }
fn default_max_retries()                    -> u32 { 2 }
fn default_retry_delay_ms()                 -> u64 { 500 }
fn default_request_timeout_secs()           -> u64 { 300 }
fn default_claude_code_max_concurrency()    -> usize { 4 }
fn default_claude_code_queue_timeout_secs() -> u64 { 30 }

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
    /// Enable Ebbinghaus-style forgetting curves for cache TTL.
    /// Popular entries (high hit_count) get their TTL extended; entries that
    /// go unaccessed for a long time expire sooner than the fixed default.
    #[serde(default = "default_true")]
    pub forgetting_enabled:          bool,
    /// How often (seconds) to run the forgetting curve adjustment sweep.
    #[serde(default = "default_forgetting_interval")]
    pub forgetting_interval_secs:    u64,
    /// Maximum TTL multiplier.  An entry with many hits cannot exceed
    /// `default_ttl_secs * forgetting_max_multiplier`.
    #[serde(default = "default_forgetting_max_multiplier")]
    pub forgetting_max_multiplier:   f64,
}

fn default_forgetting_interval()      -> u64 { 21_600 } // 6 hours
fn default_forgetting_max_multiplier() -> f64 { 8.0 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RoutingConfig {
    pub novelty_threshold:     f64,
    pub complexity_threshold:  f64,
    pub consequence_threshold: f64,
    /// Enable draft-verify: when the semantic cache has a near-miss (sim ≥
    /// draft_verify_min_sim but below embedding.sim_threshold), send the cached
    /// response as a speculative draft to the API for cheap verify+extend.
    #[serde(default = "default_true")]
    pub draft_verify_enabled:  bool,
    /// Minimum cosine similarity for a near-miss entry to qualify as a draft.
    /// Must be below [embedding] sim_threshold (full hits are served directly).
    #[serde(default = "default_draft_verify_min_sim")]
    pub draft_verify_min_sim:  f64,
}

fn default_draft_verify_min_sim() -> f64 { 0.65 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BudgetConfig {
    /// Set to false when running on a subscription plan (Pro/Max) to disable
    /// the daily spend gate.  Spend is not recorded when disabled.
    /// Hot-reloadable: change in config.toml and the gate takes effect on the
    /// next request without a restart.
    #[serde(default = "default_true")]
    pub enabled:           bool,
    pub db_path:           String,
    pub daily_limit_usd:   f64,
    pub warn_at_pct:       u8,
    pub input_per_1k_usd:  f64,
    pub output_per_1k_usd: f64,
}

fn default_true() -> bool { true }

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

// ── Limits ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LimitsConfig {
    /// Maximum requests to POST /v1/messages per minute.  0 = no limit.
    pub messages_per_minute: u32,
    /// Seconds to wait for in-flight requests to drain during graceful shutdown.
    pub shutdown_timeout_secs: u64,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        LimitsConfig {
            messages_per_minute:   30_000,
            shutdown_timeout_secs: 30,
        }
    }
}

// ── Organic learning (Layer 1: RAG few-shot injection) ────────────────────────

/// Controls the organic learning pipeline.  Layer 1 injects semantically
/// related Q&A pairs from the cache as few-shot context before each local
/// model call, so the local model answers with the benefit of prior Claude
/// responses on similar prompts.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LearningConfig {
    /// Enable few-shot context injection into local model calls.
    pub enabled:          bool,
    /// How many Q&A examples to inject per request (more = richer context,
    /// higher token cost for the local model).
    pub fewshot_k:        usize,
    /// Minimum cosine similarity for a cache entry to qualify as an example.
    /// Should be meaningfully below `embedding.sim_threshold` so we pull in
    /// related-but-not-identical prior answers.
    pub min_sim:          f64,
    /// Maximum characters of an answer to include per example.  Long answers
    /// are truncated so the injected context doesn't dominate the prompt.
    pub max_answer_chars: usize,
    /// Enable background distillation of cache entries into domain knowledge documents.
    pub distill_enabled:        bool,
    /// How often (in seconds) to run the distillation sweep across all domains.
    pub distill_interval_secs:  u64,
    /// Minimum number of live cache entries a domain needs before distillation runs.
    pub distill_min_entries:    usize,
    /// Maximum number of cache entries to feed into a single distillation call.
    pub distill_source_limit:   usize,
    /// Enable the adaptive routing threshold system (Layer 3).
    pub adapt_enabled:          bool,
    /// How often (seconds) to recompute thresholds from the routing log.
    pub adapt_interval_secs:    u64,
    /// Look-back window (seconds) for escalation rate computation.
    pub adapt_window_secs:      u64,
    /// Minimum routed samples required before adapting a (domain, intent) pair.
    pub adapt_min_samples:      usize,
    /// Escalation rate above which the novelty threshold is raised (harder to route locally).
    pub adapt_high_water:       f64,
    /// Escalation rate below which the novelty threshold is lowered (easier to route locally).
    pub adapt_low_water:        f64,
    /// How much to move the novelty threshold per adaptation step.
    pub adapt_step:             f64,
    /// How heavily explicit `![good]`/`![bad]` feedback weighs against routing-based
    /// escalation counts.  2.0 = each `![bad]` counts as 2 routing failures;
    /// each `![good]` offsets 2 failures.  Set to 0 to disable feedback weighting.
    pub adapt_feedback_weight:  f64,
    /// Enable contrastive failure learning (Layer 5).
    /// When the local model attempts a prompt but confidence is too low, the
    /// attempt and the correct API answer are stored as a contrast pair and fed
    /// into distillation so the local model learns what to avoid.
    pub contrast_enabled:       bool,
    /// Include one contrast pair per domain in few-shot injection as a labeled
    /// negative example.  Disabled by default because it adds tokens to every
    /// local model call; enable if few-shot context is too generic.
    pub contrast_in_fewshot:    bool,
    /// Max contrast pairs per domain to feed into one distillation synthesis call.
    pub contrast_source_limit:  usize,
    /// Seconds to wait before running the first distillation sweep after startup.
    /// Allows the server to accumulate cache entries before the first synthesis run.
    pub distill_warmup_secs:    u64,
    /// Enable background confidence calibration (Layer 6).
    /// Periodically replays API cache entries through the local model and measures
    /// actual accuracy vs. claimed confidence, building a per-domain correction bias.
    pub calibration_enabled:       bool,
    /// How many randomly sampled API cache entries to test per calibration run.
    pub calibration_batch_size:    usize,
    /// Look-back window (seconds) for computing calibration biases from stored samples.
    pub calibration_window_secs:   u64,
    /// Seconds between calibration runs.  Minimum 300 (5 minutes).
    #[serde(default = "default_calibration_interval")]
    pub calibration_interval_secs: u64,
}

fn default_calibration_interval() -> u64 { 3_600 } // 1 hour

impl Default for LearningConfig {
    fn default() -> Self {
        LearningConfig {
            enabled:               true,
            fewshot_k:             3,
            min_sim:               0.65,
            max_answer_chars:      1500,
            distill_enabled:       true,
            distill_interval_secs: 3600,
            distill_min_entries:   10,
            distill_source_limit:  20,
            adapt_enabled:         true,
            adapt_interval_secs:   900,   // check every 15 minutes
            adapt_window_secs:     86400, // look at last 24 hours
            adapt_min_samples:     20,
            adapt_high_water:      0.70,
            adapt_low_water:       0.25,
            adapt_step:            0.05,
            adapt_feedback_weight: 2.0,
            contrast_enabled:      true,
            contrast_in_fewshot:   false,
            contrast_source_limit:       5,
            distill_warmup_secs:         120,
            calibration_enabled:         true,
            calibration_batch_size:      20,
            calibration_window_secs:     604_800, // 7 days
            calibration_interval_secs:   3_600,   // 1 hour
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

    /// Seconds to wait after startup before bootstrapping peer discovery.
    /// Gives the server time to start accepting connections before peers try to reach us.
    #[serde(default = "default_bootstrap_delay")]
    pub bootstrap_delay_secs: u64,

    /// Seconds to wait after startup before announcing to the CNC node.
    /// Ensures the local listener is bound and ready before the CNC tries to call back.
    #[serde(default = "default_cnc_announce_delay")]
    pub cnc_announce_delay_secs: u64,
}

fn default_bootstrap_delay()    -> u64 { 5 }
fn default_cnc_announce_delay() -> u64 { 3 }

impl Default for NodeConfig {
    fn default() -> Self {
        NodeConfig {
            role:                 NodeRole::Client,
            auto_promote_peers:   false,
            cnc_url:              String::new(),
            cnc_node_id:          String::new(),
            bootstrap_delay_secs: default_bootstrap_delay(),
            cnc_announce_delay_secs: default_cnc_announce_delay(),
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

    /// Validate configuration values, returning a human-readable error if any
    /// setting is logically impossible or out of safe operating range.
    pub fn validate(&self) -> Result<()> {
        let r = &self.routing;
        if !(0.0..=1.0).contains(&r.novelty_threshold) {
            anyhow::bail!("[routing] novelty_threshold must be in [0.0, 1.0], got {}", r.novelty_threshold);
        }
        if !(0.0..=1.0).contains(&r.complexity_threshold) {
            anyhow::bail!("[routing] complexity_threshold must be in [0.0, 1.0], got {}", r.complexity_threshold);
        }
        if !(0.0..=1.0).contains(&r.consequence_threshold) {
            anyhow::bail!("[routing] consequence_threshold must be in [0.0, 1.0], got {}", r.consequence_threshold);
        }
        if r.draft_verify_enabled && !(0.0..=1.0).contains(&r.draft_verify_min_sim) {
            anyhow::bail!("[routing] draft_verify_min_sim must be in [0.0, 1.0], got {}", r.draft_verify_min_sim);
        }

        let e = &self.embedding;
        if e.enabled && !(0.0..=1.0).contains(&e.sim_threshold) {
            anyhow::bail!("[embedding] sim_threshold must be in [0.0, 1.0], got {}", e.sim_threshold);
        }
        if e.enabled && e.dimensions == 0 {
            anyhow::bail!("[embedding] dimensions must be > 0");
        }
        if e.enabled && e.dimensions > 4096 {
            anyhow::bail!(
                "[embedding] dimensions must be <= 4096 to avoid excessive memory usage, got {}",
                e.dimensions
            );
        }

        if r.draft_verify_enabled && e.enabled && r.draft_verify_min_sim >= e.sim_threshold {
            anyhow::bail!(
                "[routing] draft_verify_min_sim ({}) must be < [embedding] sim_threshold ({})",
                r.draft_verify_min_sim, e.sim_threshold
            );
        }

        let l = &self.learning;
        if l.min_sim >= e.sim_threshold && e.enabled {
            anyhow::bail!(
                "[learning] min_sim ({}) must be less than [embedding] sim_threshold ({}) \
                 or the few-shot candidate window will always be empty",
                l.min_sim, e.sim_threshold
            );
        }
        if l.adapt_high_water <= l.adapt_low_water {
            anyhow::bail!(
                "[learning] adapt_high_water ({}) must exceed adapt_low_water ({})",
                l.adapt_high_water, l.adapt_low_water
            );
        }
        if !(0.0..=1.0).contains(&l.adapt_high_water) {
            anyhow::bail!("[learning] adapt_high_water must be in [0.0, 1.0], got {}", l.adapt_high_water);
        }
        if !(0.0..=1.0).contains(&l.adapt_low_water) {
            anyhow::bail!("[learning] adapt_low_water must be in [0.0, 1.0], got {}", l.adapt_low_water);
        }

        let loc = &self.local;
        if loc.enabled && !(0.0..=1.0).contains(&loc.confidence_floor) {
            anyhow::bail!("[local] confidence_floor must be in [0.0, 1.0], got {}", loc.confidence_floor);
        }

        if self.budget.enabled && self.budget.daily_limit_usd <= 0.0 {
            anyhow::bail!("[budget] daily_limit_usd must be > 0 when budget is enabled");
        }

        if self.server.port == 0 {
            anyhow::bail!("[server] port must be non-zero");
        }

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
                host: "127.0.0.1".into(),
                port: 3000,
            },
            api: ApiConfig {
                enabled:                        true,
                model:                          "claude-sonnet-4-6".into(),
                base_url:                       "https://api.anthropic.com".into(),
                backend:                        "anthropic".into(),
                max_retries:                    2,
                retry_delay_ms:                 500,
                request_timeout_secs:           300,
                claude_code_max_concurrency:    4,
                claude_code_queue_timeout_secs: 30,
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
                db_path:                  "claude-cache.db".into(),
                max_size_mb:              500,
                default_ttl_secs:         604_800, // 7 days
                domain_ttl:               HashMap::new(),
                forgetting_enabled:       true,
                forgetting_interval_secs: 21_600,
                forgetting_max_multiplier: 8.0,
            },
            routing: RoutingConfig {
                novelty_threshold:     0.60,
                complexity_threshold:  0.40,
                consequence_threshold: 0.30,
                draft_verify_enabled:  true,
                draft_verify_min_sim:  0.65,
            },
            budget: BudgetConfig {
                enabled:           true,
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
            limits: LimitsConfig::default(),
            learning: LearningConfig::default(),
        }
    }
}
