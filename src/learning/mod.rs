use std::sync::Arc;
use anyhow::Result;
use arc_swap::ArcSwap;
use tracing::{debug, info, warn};

use crate::backend::{MessageContent, Message, MessagesRequest, ModelBackend};
use crate::cache::{CacheStore, ContrastPair};
use crate::config::AppConfig;
use crate::federation::{FederationClient, PeerKnowledge};

/// In-memory map from `(domain, intent)` to an overridden novelty threshold.
/// Populated at startup from the DB and updated by `ThresholdAdaptor` after
/// each adaptation pass.  The Router reads this atomically on every request.
pub type ThresholdMap = std::collections::HashMap<(String, String), f64>;

/// In-memory map from `(domain, intent)` to a confidence correction bias.
/// `bias = mean(actual_similarity − claimed_confidence)` over recent calibration
/// samples.  Negative bias ⟹ model overclaims; positive ⟹ underconfident.
/// Applied as `effective_conf = claimed_conf + bias` before comparing to the floor.
pub type CalibrationMap = std::collections::HashMap<(String, String), f64>;

// ── Distiller ──────────────────────────────────────────────────────────────

/// Background worker that periodically synthesizes high-hit cache entries for
/// each domain into a compact knowledge document, stored in `domain_knowledge`.
///
/// On each tick:
/// 1. Find domains with enough live entries (`distill_min_entries`).
/// 2. Pull the top `distill_source_limit` entries (ranked by hit count).
/// 3. Ask the local Ollama model to synthesize them into a reference document.
/// 4. Upsert the document — `try_local` in the router injects it as a system
///    prompt prefix so the local model answers with accumulated domain context.
#[derive(Clone)]
pub struct Distiller {
    cache:      Arc<CacheStore>,
    cfg:        Arc<ArcSwap<AppConfig>>,
    local:      Arc<dyn ModelBackend>,
    federation: Option<Arc<FederationClient>>,
}

impl Distiller {
    pub fn new(
        cache: Arc<CacheStore>,
        cfg:   Arc<ArcSwap<AppConfig>>,
        local: Arc<dyn ModelBackend>,
    ) -> Self {
        Distiller { cache, cfg, local, federation: None }
    }

    /// Attach a federation client so the distiller can fetch peer knowledge docs
    /// and blend mesh-wide learning into local synthesis runs.
    pub fn with_federation(mut self, fed: Arc<FederationClient>) -> Self {
        self.federation = Some(fed);
        self
    }

    /// Spawn the background distillation loop.  Fires after a configurable warmup
    /// delay so startup traffic can seed the cache before the first sweep.
    pub async fn run(self) {
        let warmup = self.cfg.load().learning.distill_warmup_secs;
        tokio::time::sleep(std::time::Duration::from_secs(warmup)).await;

        loop {
            let cfg = self.cfg.load();
            if !cfg.learning.distill_enabled {
                tokio::time::sleep(std::time::Duration::from_secs(
                    cfg.learning.distill_interval_secs.max(60),
                )).await;
                continue;
            }
            let interval = cfg.learning.distill_interval_secs.max(60);
            drop(cfg);

            match self.sweep().await {
                Ok(n) if n > 0 => info!("distillation: updated {n} domain knowledge doc(s)"),
                Ok(_)          => debug!("distillation: no domains ready yet"),
                Err(e)         => warn!("distillation sweep error: {e}"),
            }

            tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        }
    }

    /// Run one distillation sweep: check all domains and (re)distill those that
    /// have enough entries.  Returns the number of documents updated.
    pub async fn sweep(&self) -> Result<usize> {
        let cfg = self.cfg.load();
        let min = cfg.learning.distill_min_entries as i64;

        let domains = self.cache.distillation_candidates(min).await?;
        let mut updated = 0usize;

        for domain in &domains {
            match self.distill_domain(domain).await {
                Ok(doc) => {
                    info!("distilled '{domain}': {} chars", doc.len());
                    updated += 1;
                }
                Err(e) => warn!("distillation failed for '{domain}': {e}"),
            }
        }
        Ok(updated)
    }

    /// Manually distill a specific domain and return the generated document.
    /// Callable from the management endpoint.
    pub async fn distill_domain(&self, domain: &str) -> Result<String> {
        let cfg     = self.cfg.load();
        let entries = self.cache
            .top_entries_for_domain(domain, cfg.learning.distill_source_limit as i64)
            .await?;

        if entries.is_empty() {
            anyhow::bail!("no entries for domain '{domain}'");
        }

        // Layer 5: fetch contrast pairs to include as negative examples.
        let contrasts = if cfg.learning.contrast_enabled {
            self.cache
                .contrast_pairs_for_domain(domain, cfg.learning.contrast_source_limit as i64)
                .await
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Feature 4: fetch peer knowledge docs and contrast pairs from the federation
        // mesh and blend them into the synthesis prompt so the distilled document
        // reflects collective learning, not just this node's cache history.
        let peer_knowledge: Vec<PeerKnowledge> = if cfg.federation.enabled {
            if let Some(ref fed) = self.federation {
                let pk = fed.fetch_peer_knowledge(domain).await;
                if !pk.is_empty() {
                    info!("distillation: blending knowledge from {} peer(s) for '{domain}'", pk.len());
                }
                pk
            } else { vec![] }
        } else { vec![] };

        let prompt_body = build_distillation_prompt(domain, &entries, &contrasts, &peer_knowledge);
        let req = synthesis_request(&cfg, prompt_body);
        let timeout_secs = cfg.local.timeout_secs;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            self.local.complete(&req),
        ).await
        .map_err(|_| anyhow::anyhow!(
            "distillation synthesis timed out after {timeout_secs}s for domain '{domain}'"
        ))??;
        let doc    = result.response.text_content();

        if doc.is_empty() {
            anyhow::bail!("local model returned empty synthesis for '{domain}'");
        }

        self.cache.store_knowledge_doc(domain, &doc, entries.len()).await?;
        Ok(doc)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn build_distillation_prompt(
    domain:    &str,
    entries:   &[(String, String)],
    contrasts: &[ContrastPair],
    peers:     &[PeerKnowledge],
) -> String {
    let mut body = format!(
        "Here are {} representative Q&A pairs from the '{}' programming domain \
         (ranked by how often they were used):\n",
        entries.len(), domain
    );

    for (i, (q, a)) in entries.iter().enumerate() {
        let answer = truncate_chars(a, 600);
        body.push_str(&format!("\n--- Example {} ---\nQ: {}\nA: {}\n", i + 1, q, answer));
    }

    // Layer 5: include local contrast pairs as labeled failure examples.
    if !contrasts.is_empty() {
        body.push_str(&format!(
            "\nHere are {} cases where a previous local model answer was INCORRECT \
             (confidence too low, escalated to the authoritative API). Study these to \
             understand what patterns to avoid:\n",
            contrasts.len()
        ));
        for (i, pair) in contrasts.iter().enumerate() {
            let wrong   = truncate_chars(&pair.local_attempt,  400);
            let correct = truncate_chars(&pair.correct_answer, 400);
            body.push_str(&format!(
                "\n--- Contrast {} (intent: {}) ---\n[INCORRECT]: {}\n[CORRECT]:   {}\n",
                i + 1, pair.intent, wrong, correct
            ));
        }
    }

    // Feature 4: blend in peer knowledge docs and peer contrast pairs from the
    // federation mesh so the synthesized document reflects collective learning.
    let peer_docs: Vec<&str> = peers.iter()
        .filter_map(|pk| pk.knowledge_doc.as_deref())
        .filter(|doc| !doc.is_empty())
        .collect();

    if !peer_docs.is_empty() {
        body.push_str(&format!(
            "\n\nDistilled knowledge from {} peer node(s) in the federation mesh \
             (use as supplementary context, weighted equally with local knowledge):\n",
            peer_docs.len()
        ));
        for (i, doc) in peer_docs.iter().enumerate() {
            let snippet = truncate_chars(doc, 400);
            body.push_str(&format!("\n--- Peer {} Knowledge ---\n{}\n", i + 1, snippet));
        }
    }

    let peer_contrasts: Vec<&crate::federation::PeerContrastPair> = peers.iter()
        .flat_map(|pk| pk.contrast_pairs.iter())
        .collect();

    if !peer_contrasts.is_empty() {
        body.push_str(&format!(
            "\n\nAdditional failure cases from peer node(s) — patterns observed in the \
             mesh to avoid:\n"
        ));
        for (i, pair) in peer_contrasts.iter().enumerate().take(5) {
            let wrong   = truncate_chars(&pair.wrong,   300);
            let correct = truncate_chars(&pair.correct, 300);
            body.push_str(&format!(
                "\n--- Peer Contrast {} (intent: {}) ---\n[INCORRECT]: {}\n[CORRECT]:   {}\n",
                i + 1, pair.intent, wrong, correct
            ));
        }
    }

    let has_contrasts = !contrasts.is_empty() || !peer_contrasts.is_empty();

    body.push_str(&format!(
        "\nSynthesize the key patterns, idioms, conventions, and knowledge from \
         these examples into a compact '{}' reference document. Focus on: \
         recurring patterns, common solutions, domain-specific conventions, and \
         the user's apparent style and preferences.",
        domain
    ));

    if has_contrasts {
        body.push_str(
            " Also explicitly note any recurring mistake patterns from the contrast \
             examples above and what the correct approach should be instead."
        );
    }

    body.push_str(
        " Keep it under 700 words. Write in a clear, factual style — this document \
         will be injected as background context before future answers in this domain."
    );
    body
}

/// Truncate `s` to at most `max_bytes` bytes, aligned to a valid UTF-8 char boundary.
fn truncate_chars(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes { return s; }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) { end -= 1; }
    &s[..end]
}

fn synthesis_request(cfg: &AppConfig, prompt_body: String) -> MessagesRequest {
    MessagesRequest {
        model:          cfg.local.model_id.clone(),
        messages:       vec![Message {
            role:    "user".to_string(),
            content: MessageContent::Text(prompt_body),
        }],
        max_tokens:     1024,
        system:         None,
        stream:         Some(false),
        tools:          None,
        extra:          Default::default(),
        anthropic_beta: None,
    }
}

// ── ThresholdAdaptor ───────────────────────────────────────────────────────

/// Watches the routing log and dynamically adjusts the per-(domain, intent)
/// novelty threshold so the routing gate self-calibrates as the local model
/// learns from Layers 1 and 2.
///
/// High escalation rate (>70%): raise the novelty threshold — the local model
/// isn't ready for this domain yet, send more traffic to Claude.
///
/// Low escalation rate (<25%): lower the threshold — the local model is
/// handling this domain confidently, route more traffic locally.
#[derive(Clone)]
pub struct ThresholdAdaptor {
    cache:      Arc<CacheStore>,
    cfg:        Arc<ArcSwap<AppConfig>>,
    thresholds: Arc<ArcSwap<ThresholdMap>>,
}

impl ThresholdAdaptor {
    pub fn new(
        cache:      Arc<CacheStore>,
        cfg:        Arc<ArcSwap<AppConfig>>,
        thresholds: Arc<ArcSwap<ThresholdMap>>,
    ) -> Self {
        ThresholdAdaptor { cache, cfg, thresholds }
    }

    /// Run the background adaptation loop.  Loads persisted overrides from the
    /// DB on startup so thresholds survive a proxy restart.
    pub async fn run(self) {
        // Load any previously computed overrides so routing is calibrated
        // immediately on restart rather than starting cold.
        match self.cache.load_threshold_overrides().await {
            Ok(map) if !map.is_empty() => {
                info!("threshold adaptor: loaded {} override(s) from DB", map.len());
                self.thresholds.store(Arc::new(map));
            }
            Ok(_)  => {}
            Err(e) => warn!("threshold adaptor: failed to load overrides: {e}"),
        }

        loop {
            let cfg = self.cfg.load();
            let interval = cfg.learning.adapt_interval_secs.max(60);
            drop(cfg);

            tokio::time::sleep(std::time::Duration::from_secs(interval)).await;

            let cfg = self.cfg.load();
            if !cfg.learning.adapt_enabled { continue; }
            drop(cfg);

            match self.adapt().await {
                Ok(n) if n > 0 => info!("threshold adaptor: adjusted {n} domain/intent pair(s)"),
                Ok(_)          => debug!("threshold adaptor: all domains within acceptable band"),
                Err(e)         => warn!("threshold adaptor error: {e}"),
            }
        }
    }

    /// One adaptation pass: compute escalation rates, blend quality feedback,
    /// and update per-(domain, intent) novelty threshold overrides.
    pub async fn adapt(&self) -> Result<usize> {
        let cfg         = self.cfg.load();
        let window      = cfg.learning.adapt_window_secs as i64;
        let min_samples = cfg.learning.adapt_min_samples as i64;
        let high_water  = cfg.learning.adapt_high_water;
        let low_water   = cfg.learning.adapt_low_water;
        let step        = cfg.learning.adapt_step;
        let base        = cfg.routing.novelty_threshold;
        let fw          = cfg.learning.adapt_feedback_weight;

        let esc_stats  = self.cache.escalation_stats(window, min_samples).await?;
        let qual_stats = self.cache.quality_stats(window).await?;

        // Build quality lookup: (domain, intent) → (bad_count, good_count)
        let qual_map: std::collections::HashMap<(&str, &str), (i64, i64)> = qual_stats.iter()
            .map(|s| ((s.domain.as_str(), s.intent.as_str()), (s.bad_count, s.good_count)))
            .collect();

        let mut current: ThresholdMap = (**self.thresholds.load()).clone();
        let mut changed = 0usize;

        for stat in &esc_stats {
            let key       = (stat.domain.clone(), stat.intent.clone());
            let current_t = current.get(&key).copied().unwrap_or(base);

            // Blend explicit feedback into the escalation rate:
            //   bad  signals count as weighted failures
            //   good signals offset them (lower effective escalation)
            let (bad, good) = qual_map
                .get(&(stat.domain.as_str(), stat.intent.as_str()))
                .copied()
                .unwrap_or((0, 0));

            // Each `![bad]` adds fw weighted failures; each `![good]` subtracts fw
            // from the failure count so good feedback actively lowers the rate, not
            // merely dilutes it.
            let base_esc  = stat.escalation_rate * stat.sample_count as f64;
            let adj_esc   = (base_esc + bad as f64 * fw - good as f64 * fw).max(0.0);
            let adj_total = stat.sample_count as f64 + (bad as f64 + good as f64) * fw;
            let rate      = if adj_total > 0.0 { adj_esc / adj_total }
                            else { stat.escalation_rate };

            let new_t = if rate > high_water {
                Some((current_t + step).min(0.95))
            } else if rate < low_water {
                Some((current_t - step).max(base * 0.70))
            } else {
                None
            };

            if let Some(t) = new_t {
                if (t - current_t).abs() > 1e-6 {
                    info!(
                        "adapt {}/{}: esc={:.0}% bad={} good={} → adj={:.0}% → novelty_t {:.2}→{:.2} (n={})",
                        stat.domain, stat.intent,
                        stat.escalation_rate * 100.0, bad, good, rate * 100.0,
                        current_t, t, stat.sample_count
                    );
                    current.insert(key, t);
                    self.cache.store_threshold_override(
                        &stat.domain, &stat.intent, t,
                        rate, stat.sample_count,
                    ).await?;
                    changed += 1;
                }
            }
        }

        // Feedback-only pass: adjust domains that have quality signals but not
        // enough routing data to meet min_samples.  Requires ≥3 combined signals.
        let esc_keys: std::collections::HashSet<_> = esc_stats.iter()
            .map(|s| (s.domain.as_str(), s.intent.as_str()))
            .collect();

        for stat in &qual_stats {
            if esc_keys.contains(&(stat.domain.as_str(), stat.intent.as_str())) {
                continue; // already handled in the main pass above
            }
            let total = stat.bad_count + stat.good_count;
            if total < 3 { continue; }

            let key       = (stat.domain.clone(), stat.intent.clone());
            let current_t = current.get(&key).copied().unwrap_or(base);
            let rate      = stat.bad_count as f64 / total as f64;

            let new_t = if rate > high_water {
                Some((current_t + step).min(0.95))
            } else if rate < low_water {
                Some((current_t - step).max(base * 0.70))
            } else {
                None
            };

            if let Some(t) = new_t {
                if (t - current_t).abs() > 1e-6 {
                    info!(
                        "adapt {}/{}: feedback-only bad={} good={} → rate={:.0}% → novelty_t {:.2}→{:.2}",
                        stat.domain, stat.intent,
                        stat.bad_count, stat.good_count, rate * 100.0,
                        current_t, t
                    );
                    current.insert(key, t);
                    self.cache.store_threshold_override(
                        &stat.domain, &stat.intent, t,
                        rate, total,
                    ).await?;
                    changed += 1;
                }
            }
        }

        if changed > 0 {
            self.thresholds.store(Arc::new(current));
        }
        Ok(changed)
    }
}

// ── ForgettingCurveWorker ─────────────────────────────────────────────────

/// Background worker that periodically applies Ebbinghaus-style forgetting
/// curves to cache TTLs.
///
/// Entries that keep being accessed have their `expires_at` pushed forward
/// (proportional to `1 + ln(1 + hit_count)`).  Entries that go stale have
/// their `expires_at` anchored to the old `last_hit_at` timestamp, so they
/// naturally expire sooner than a flat TTL scheme would allow.
#[derive(Clone)]
pub struct ForgettingCurveWorker {
    cache: Arc<CacheStore>,
    cfg:   Arc<ArcSwap<AppConfig>>,
}

impl ForgettingCurveWorker {
    pub fn new(cache: Arc<CacheStore>, cfg: Arc<ArcSwap<AppConfig>>) -> Self {
        ForgettingCurveWorker { cache, cfg }
    }

    pub async fn run(self) {
        loop {
            let interval = self.cfg.load().cache.forgetting_interval_secs.max(3600);
            tokio::time::sleep(std::time::Duration::from_secs(interval)).await;

            let cfg = self.cfg.load();
            if !cfg.cache.forgetting_enabled { continue; }
            let default_ttl = cfg.cache.default_ttl_secs;
            let domain_ttls = cfg.cache.domain_ttl.clone();
            let max_mult    = cfg.cache.forgetting_max_multiplier;
            drop(cfg);

            match self.cache.adjust_ttl_forgetting(default_ttl, &domain_ttls, max_mult).await {
                Ok(n) if n > 0 => info!("forgetting curves: adjusted TTL for {n} cache entr(ies)"),
                Ok(_)          => debug!("forgetting curves: no significant TTL changes needed"),
                Err(e)         => warn!("forgetting curves error: {e}"),
            }
        }
    }
}

// ── CalibrationRunner ──────────────────────────────────────────────────────

/// Background worker (Layer 6) that measures how well the local model's
/// self-reported confidence actually predicts answer quality.
///
/// Each hourly run:
/// 1. Randomly samples a batch of API-origin cache entries.
/// 2. Re-runs each prompt through the local model.
/// 3. Computes word-overlap similarity between the local and API answers.
/// 4. Records `(claimed_conf, actual_sim)` in `calibration_log`.
/// 5. Recomputes per-(domain, intent) bias = mean(actual − claimed) and
///    swaps the in-memory `CalibrationMap` so the router sees it immediately.
#[derive(Clone)]
pub struct CalibrationRunner {
    cache:  Arc<CacheStore>,
    cfg:    Arc<ArcSwap<AppConfig>>,
    local:  Arc<dyn ModelBackend>,
    biases: Arc<ArcSwap<CalibrationMap>>,
}

impl CalibrationRunner {
    pub fn new(
        cache:  Arc<CacheStore>,
        cfg:    Arc<ArcSwap<AppConfig>>,
        local:  Arc<dyn ModelBackend>,
        biases: Arc<ArcSwap<CalibrationMap>>,
    ) -> Self {
        CalibrationRunner { cache, cfg, local, biases }
    }

    /// Warm up from persisted samples, then run on a configurable interval.
    pub async fn run(self) {
        // Load persisted biases immediately so routing is calibrated before the
        // first batch runs.
        let window = self.cfg.load().learning.calibration_window_secs as i64;
        match self.cache.load_calibration_biases(window).await {
            Ok(map) if !map.is_empty() => {
                info!("calibration: loaded {} bias(es) from DB", map.len());
                self.biases.store(Arc::new(map));
            }
            Ok(_)  => {}
            Err(e) => warn!("calibration: failed to load initial biases: {e}"),
        }

        // Wait one full interval before the first batch so startup traffic can
        // seed the cache before we replay entries (consistent with Distiller's
        // warmup pattern; avoids 20 local-model calls right at boot).
        let initial_interval = self.cfg.load().learning.calibration_interval_secs.max(300);
        tokio::time::sleep(std::time::Duration::from_secs(initial_interval)).await;

        loop {
            let cfg = self.cfg.load();
            if !cfg.learning.enabled || !cfg.learning.calibration_enabled {
                let interval = cfg.learning.calibration_interval_secs.max(300);
                drop(cfg);
                tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
                continue;
            }
            let batch    = cfg.learning.calibration_batch_size;
            let window   = cfg.learning.calibration_window_secs as i64;
            let interval = cfg.learning.calibration_interval_secs.max(300);
            drop(cfg);

            match self.run_batch(batch, window).await {
                Ok(n) if n > 0 => info!("calibration: sampled {n} entr(ies), biases updated"),
                Ok(_)          => debug!("calibration: no API entries to sample yet"),
                Err(e)         => warn!("calibration run error: {e}"),
            }

            tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        }
    }

    async fn run_batch(&self, batch: usize, window_secs: i64) -> Result<usize> {
        let entries = self.cache.sample_api_entries(batch).await?;
        if entries.is_empty() { return Ok(0); }

        let mut sampled = 0usize;
        for entry in &entries {
            let req = MessagesRequest {
                model:          String::new(),
                messages:       vec![Message {
                    role:    "user".to_string(),
                    content: MessageContent::Text(entry.prompt_text.clone()),
                }],
                max_tokens:     512,
                system:         None,
                stream:         Some(false),
                tools:          None,
                extra:          Default::default(),
                anthropic_beta: None,
            };

            let result = match self.local.complete(&req).await {
                Ok(r)  => r,
                Err(e) => { debug!("calibration local call skipped: {e}"); continue; }
            };

            let claimed = result.confidence.unwrap_or(0.5).clamp(0.0, 1.0);
            let local_text  = result.response.text_content();
            let cached_text = extract_text_from_response_json(&entry.response);

            if local_text.is_empty() || cached_text.is_empty() { continue; }

            let actual_sim = word_jaccard(&local_text, &cached_text);
            let _ = self.cache.store_calibration_sample(
                &entry.domain, &entry.intent, claimed, actual_sim,
            ).await;
            sampled += 1;
        }

        if sampled > 0 {
            if let Ok(map) = self.cache.load_calibration_biases(window_secs).await {
                if !map.is_empty() {
                    debug!("calibration: {} domain/intent bias(es) recomputed", map.len());
                    self.biases.store(Arc::new(map));
                }
            }
        }
        Ok(sampled)
    }
}

// ── Calibration helpers ────────────────────────────────────────────────────

fn extract_text_from_response_json(json: &str) -> String {
    serde_json::from_str::<crate::backend::MessagesResponse>(json)
        .map(|r| r.text_content())
        .unwrap_or_default()
}

/// Jaccard word overlap — fast similarity proxy that works well for comparing
/// technical answers without needing an embedder on the calibration path.
fn word_jaccard(a: &str, b: &str) -> f64 {
    use std::collections::HashSet;
    let wa: HashSet<&str> = a.split_whitespace().collect();
    let wb: HashSet<&str> = b.split_whitespace().collect();
    let intersection = wa.intersection(&wb).count();
    let union        = wa.union(&wb).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}
