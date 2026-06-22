use anyhow::Result;
use arc_swap::ArcSwap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::backend::{BackendResult, FewShotExample, MessagesRequest, MessagesResponse, ModelBackend};
use crate::budget::BudgetLedger;
use crate::cache::CacheStore;
use crate::config::AppConfig;
use crate::domain;
use crate::embedding::Embedder;
use crate::error::ProxyError;
use crate::federation::FederationClient;
use crate::learning::{CalibrationMap, ThresholdMap};
use crate::policy;
use crate::scoring;

/// Where the response came from.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteDecision {
    ExactCache,
    SemanticCache,
    FederationPeer(String),
    LocalModel,
    Api,
}

impl RouteDecision {
    pub fn as_str(&self) -> &str {
        match self {
            RouteDecision::ExactCache      => "exact_cache",
            RouteDecision::SemanticCache   => "semantic_cache",
            RouteDecision::FederationPeer(_) => "federation",
            RouteDecision::LocalModel      => "local",
            RouteDecision::Api             => "api",
        }
    }
}

pub struct RoutedResponse {
    pub response:   MessagesResponse,
    pub decision:   RouteDecision,
    pub latency_ms: u64,
    pub saved_usd:  f64,
    pub trace:      RouteTrace,
}

/// Per-request decision audit trail populated as the routing pipeline runs.
/// Each field represents one factor that influenced the routing outcome.
#[derive(Debug, Default, Clone)]
pub struct RouteTrace {
    pub domain:            String,
    pub intent:            String,
    // Layer 1: few-shot examples injected
    pub l1_shots:          usize,
    pub l1_min_sim:        f64,
    pub l1_max_sim:        f64,
    // Layer 2: distilled domain knowledge doc
    pub l2_doc:            bool,
    pub l2_doc_chars:      usize,
    // Layer 3: adaptive threshold in effect
    pub l3_threshold:      f64,   // effective value (adaptive override or config base)
    pub l3_base:           f64,   // raw config value
    pub l3_adapted:        bool,  // true when adaptive override differed from config
    // Layer 5: contrast pair injected into few-shot
    pub l5_contrast:       bool,
    // Routing gate scores
    pub novelty_score:     f64,
    pub complexity_score:  f64,
    pub consequence_score: f64,
    // Local model outcome
    pub confidence:        Option<f64>,
    // Why the routing gate sent this to API (absent on cache hits)
    pub miss_reason:       Option<String>,
}

#[derive(Clone)]
pub struct Router {
    cfg:              Arc<ArcSwap<AppConfig>>,
    cache:            Arc<CacheStore>,
    budget:           Arc<BudgetLedger>,
    embedder:         Arc<dyn Embedder>,
    local:            Arc<dyn ModelBackend>,
    api:              Arc<dyn ModelBackend>,
    federation:       Option<Arc<FederationClient>>,
    thresholds:       Arc<ArcSwap<ThresholdMap>>,
    calibration:      Arc<ArcSwap<CalibrationMap>>,
    /// Unix timestamp until which the API backend is self-disabled after a
    /// no_api_access failure.  0 = not disabled.  Expires automatically so
    /// credential rotation or a subscription change recovers without restart.
    api_disabled_until: Arc<AtomicU64>,
}

impl Router {
    pub fn new(
        cfg:         Arc<ArcSwap<AppConfig>>,
        cache:       Arc<CacheStore>,
        budget:      Arc<BudgetLedger>,
        embedder:    Arc<dyn Embedder>,
        local:       Arc<dyn ModelBackend>,
        api:         Arc<dyn ModelBackend>,
        thresholds:  Arc<ArcSwap<ThresholdMap>>,
        calibration: Arc<ArcSwap<CalibrationMap>>,
    ) -> Self {
        Router {
            cfg, cache, budget, embedder, local, api,
            federation: None, thresholds, calibration,
            api_disabled_until: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Attach a federation client.  Call after `new()` before first use.
    pub fn with_federation(mut self, fed: Arc<FederationClient>) -> Self {
        self.federation = Some(fed);
        self
    }

    /// Soft-disable the API backend for 5 minutes after a no_api_access failure.
    /// Unlike patching `api.enabled`, this expires automatically — so a credential
    /// rotation (e.g. OAuth token refresh) or a subscription-tier change recovers
    /// without restarting the proxy.  Only a hard `api.enabled = false` in
    /// config.toml permanently suppresses API calls.
    fn disable_api_backend(&self, reason: &str) {
        warn!(
            "API access denied: {reason}. \
             Serving from cache and local model only for the next 5 minutes. \
             For Claude Pro: verify your subscription includes API access. \
             For API keys: check billing at console.anthropic.com."
        );
        let until = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_add(300); // 5-minute cooldown
        self.api_disabled_until.store(until, Ordering::Relaxed);
    }

    /// Returns true when the API is either hard-disabled in config or
    /// soft-disabled (self-healing cooldown after a no_api_access failure).
    fn api_blocked(&self, cfg: &crate::config::AppConfig) -> bool {
        if !cfg.api.enabled { return true; }
        let until = self.api_disabled_until.load(Ordering::Relaxed);
        if until == 0 { return false; }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now < until
    }

    /// Compute an embedding for `text` using the router's embedder, or return
    /// `None` if embedding is disabled or the call fails.  Used by callers that
    /// want to pre-compute the embedding so `cache_streamed` can skip it.
    pub async fn embed_if_enabled(&self, text: &str) -> Option<Vec<f32>> {
        if !self.cfg.load().embedding.enabled { return None; }
        self.embedder.embed(text).await.ok()
    }

    pub async fn route(&self, req: &MessagesRequest) -> Result<RoutedResponse> {
        let start = Instant::now();
        let cfg   = self.cfg.load();

        // ── Annotation pre-pass ───────────────────────────────────────────────
        // Extract all proxy annotations before routing so they never reach the
        // upstream model.
        //   ![good] / ![bad]  — strip and record quality feedback for the adaptor
        //   ![direct]         — skip cache and local model entirely
        let feedback  = req.extract_feedback_annotation();
        let direct    = req.has_direct_annotation();

        let stripped_owned;
        let req = if feedback.is_some() || direct {
            stripped_owned = req.strip_all_annotations();
            &stripped_owned
        } else {
            req
        };

        let prompt = req.prompt_text();

        // Record explicit quality feedback (fire-and-forget; classify after stripping)
        if let Some(signal) = feedback {
            let shape = domain::classify(&prompt);
            if let Err(e) = self.cache
                .record_feedback(&shape.domain, &shape.intent, signal.as_str(), "explicit")
                .await
            {
                warn!("failed to record quality feedback: {e}");
            } else {
                debug!("feedback: {} for {}/{}", signal.as_str(), shape.domain, shape.intent);
            }
        }

        // Detect implicit quality feedback from conversation continuation patterns.
        // Looks for contradiction / affirmation markers in the current user message
        // relative to the previous assistant turn, classified against the PRIOR
        // user prompt's domain so the signal lands on the right (domain, intent).
        // Skip when an explicit annotation was already recorded — avoids double-counting
        // the same interaction (e.g. "![good] thanks" recording two Good signals).
        if feedback.is_none() {
            if let Some((signal, prior_prompt)) = req.detect_implicit_feedback() {
                let prior_shape = domain::classify(&prior_prompt);
                if let Err(e) = self.cache
                    .record_feedback(&prior_shape.domain, &prior_shape.intent, signal.as_str(), "implicit")
                    .await
                {
                    warn!("failed to record implicit feedback: {e}");
                } else {
                    debug!("implicit feedback: {} for {}/{}", signal.as_str(), prior_shape.domain, prior_shape.intent);
                }
            }
        }

        // ── Step 0: fast-path for tools or ![direct] annotation ──────────────
        // Tool-use requires live API semantics; !direct is an explicit user override.
        if req.has_tools() || direct {
            debug!("{} → api", if direct { "![direct]" } else { "tool-use fast-path" });
            let shape = domain::classify(&prompt);
            let pol   = policy::infer(&shape, &prompt, &cfg);
            return self.call_api_and_cache(req, &shape, &pol, start,
                Some(if direct { "user_direct" } else { "tool_use" }), None, None, None).await;
        }

        // ── Step 1: Classify ─────────────────────────────────────────────
        let shape = domain::classify(&prompt);
        debug!("classified: {}", shape.display());

        let mut trace = RouteTrace {
            domain: shape.domain.clone(),
            intent: shape.intent.clone(),
            ..Default::default()
        };

        // ── Step 2: Policy ───────────────────────────────────────────────
        let pol = policy::infer(&shape, &prompt, &cfg);
        if pol.bypass_cache {
            debug!("policy bypass → api");
            return self.call_api_and_cache(req, &shape, &pol, start, Some("policy_bypass"), None, None, None).await;
        }

        // ── Step 3: Exact cache ──────────────────────────────────────────
        if let Some(entry) = self.cache.lookup_exact(&prompt, req.normalized_system().as_deref()).await? {
            info!("exact cache hit: {}", &entry.id[..8]);
            let resp = serde_json::from_str(&entry.response)?;
            let latency_ms = start.elapsed().as_millis() as u64;
            let saved = estimate_api_cost(&self.budget, req).await;
            self.log(&shape, RouteDecision::ExactCache, "cache", latency_ms, None, req, saved, None, None, None).await;
            return Ok(RoutedResponse {
                response: resp,
                decision: RouteDecision::ExactCache,
                latency_ms,
                saved_usd: saved,
                trace: trace.clone(),
            });
        }

        // ── Step 3.5: Federation exact lookup ─────────────────────────────
        // Ask trusted peers if they have the exact same prompt hash.
        if let Some(ref fed) = self.federation {
            let hash = CacheStore::content_key(&prompt, req.normalized_system().as_deref());
            if let Some(fed_entry) = fed.lookup(&hash).await {
                info!("federation exact hit from {}", &fed_entry.node_id[..16.min(fed_entry.node_id.len())]);
                let resp: MessagesResponse = serde_json::from_str(&fed_entry.response)?;
                let latency_ms = start.elapsed().as_millis() as u64;
                let saved = estimate_api_cost(&self.budget, req).await;
                let pol = policy::infer(&shape, &prompt, &cfg);
                if pol.should_cache() {
                    let _ = self.cache.store(&shape, &prompt, req.normalized_system().as_deref(),
                        &fed_entry.response, "federated", None, pol.ttl_secs, false, false).await;
                }
                let node_id = fed_entry.node_id.clone();
                self.log(&shape, RouteDecision::FederationPeer(node_id.clone()), "federation", latency_ms, None, req, saved, None, None, None).await;
                return Ok(RoutedResponse {
                    response: resp,
                    decision: RouteDecision::FederationPeer(node_id),
                    latency_ms,
                    saved_usd: saved,
                    trace: trace.clone(),
                });
            }
        }

        // ── Step 4: Embedding (computed once, used for local + federation) ─
        // Use only the last user message so multi-turn conversations don't produce
        // false semantic hits (e.g. "Thank you" after a long prior turn would otherwise
        // embed nearly identically to the prior turn and serve the wrong cached response).
        let embed_text = req.last_user_text();
        let embedding = if cfg.embedding.enabled {
            match self.embedder.embed(&embed_text).await {
                Ok(emb) => Some(emb),
                Err(e)  => { warn!("embedding failed, skipping semantic lookups: {e}"); None }
            }
        } else {
            None
        };

        // ── Step 4a: Local semantic cache ─────────────────────────────────
        // Semantic lookup is skipped for multi-turn conversations. The last user
        // message alone ("What else?", "Can you elaborate?") is often meaningful
        // only in context — serving a cached response from a different session
        // would be a non-sequitur. The exact cache (step 3) still applies for
        // identical full-conversation replays. Embeddings are still computed above
        // so L1 few-shot injection and the routing gate near-miss signal work normally.
        let is_multi_turn = req.messages.iter().filter(|m| m.role == "user").count() > 1;
        let emb_model = self.embedder.model();
        let mut best_semantic_sim: Option<f64> = None;
        if let Some(ref emb) = embedding {
            if !is_multi_turn {
                let hits = self.cache.lookup_semantic(
                    &shape.domain, emb, cfg.embedding.sim_threshold, 1, emb_model,
                ).await?;
                if let Some((entry, sim)) = hits.into_iter().next() {
                    info!("semantic cache hit (sim={sim:.3}): {}", &entry.id[..8]);
                    let resp = serde_json::from_str(&entry.response)?;
                    let latency_ms = start.elapsed().as_millis() as u64;
                    let saved = estimate_api_cost(&self.budget, req).await;
                    self.log(&shape, RouteDecision::SemanticCache, "cache", latency_ms, None, req, saved, None, None, None).await;
                    return Ok(RoutedResponse {
                        response: resp,
                        decision: RouteDecision::SemanticCache,
                        latency_ms,
                        saved_usd: saved,
                        trace: trace.clone(),
                    });
                }
            }
            // No hit above threshold — probe for near-miss similarity so the
            // routing gate knows how familiar this prompt shape actually is.
            best_semantic_sim = self.cache.best_semantic_sim(&shape.domain, emb, emb_model).await.ok().flatten();
        }

        // ── Step 4b: Federation semantic lookup ───────────────────────────
        // Same multi-turn guard applies to federation peers.
        // Ask trusted peers for semantically similar entries using our embedding.
        if let (Some(ref fed), Some(ref emb)) = (&self.federation, &embedding) {
            if !is_multi_turn {
                if let Some((fed_entry, sim)) = fed.lookup_semantic(
                    emb, &shape.domain, cfg.embedding.sim_threshold, 1,
                ).await {
                    info!("federation semantic hit (sim={sim:.3}) from {}",
                        &fed_entry.node_id[..16.min(fed_entry.node_id.len())]);
                    let resp: MessagesResponse = serde_json::from_str(&fed_entry.response)?;
                    let latency_ms = start.elapsed().as_millis() as u64;
                    let saved = estimate_api_cost(&self.budget, req).await;
                    // Cache locally so future identical prompts skip the peer call.
                    let pol = policy::infer(&shape, &prompt, &cfg);
                    if pol.should_cache() {
                        if let Ok(cid) = self.cache.store(&shape, &prompt, req.normalized_system().as_deref(),
                                &fed_entry.response, "federated", None, pol.ttl_secs, false, false).await {
                            if let Some(ref emb2) = embedding {
                                let _ = self.cache.store_embedding(&cid, emb2, self.embedder.model()).await;
                            }
                        }
                    }
                    let node_id = fed_entry.node_id.clone();
                    self.log(&shape, RouteDecision::FederationPeer(node_id.clone()), "federation", latency_ms, None, req, saved, None, None, None).await;
                    return Ok(RoutedResponse {
                        response: resp,
                        decision: RouteDecision::FederationPeer(node_id),
                        latency_ms,
                        saved_usd: saved,
                        trace: trace.clone(),
                    });
                }
            }
        }

        // ── Step 5: Routing gate ──────────────────────────────────────────
        // How familiar are we with this domain+intent shape?
        let hit_count = self.cache.domain_hit_count(&shape.domain, &shape.intent)
            .await
            .unwrap_or(0);
        let score = scoring::score_prompt(&shape, &prompt, hit_count, best_semantic_sim);
        debug!("routing score: {}", score.display());

        trace.novelty_score     = score.novelty;
        trace.complexity_score  = score.complexity;
        trace.consequence_score = score.consequence;

        let r = &cfg.routing;
        // Apply the adaptive novelty threshold override for this domain/intent (Layer 3).
        // Falls back to the config value when no override has been computed yet.
        let novelty_t = {
            let map = self.thresholds.load();
            map.get(&(shape.domain.clone(), shape.intent.clone()))
               .copied()
               .unwrap_or(r.novelty_threshold)
        };
        trace.l3_base      = r.novelty_threshold;
        trace.l3_threshold = novelty_t;
        trace.l3_adapted   = (novelty_t - r.novelty_threshold).abs() > 1e-9;

        if !score.should_use_local(novelty_t, r.complexity_threshold, r.consequence_threshold) {
            let miss = score.gate_miss_reason(novelty_t, r.complexity_threshold, r.consequence_threshold);
            trace.miss_reason = Some(miss.to_string());

            if self.api_blocked(&cfg) {
                debug!("routing gate → api skipped (disabled or cooling down), trying local");
            } else {
                debug!("routing gate → api ({}) novelty_t={:.2} miss={miss}", score.display(), novelty_t);
                let sj = scores_json(&score, novelty_t);
                let api_result = self.call_api_and_cache(req, &shape, &pol, start, Some(miss), None, Some(sj), embedding.as_deref()).await;
                match api_result {
                    Ok(mut r2) => { r2.trace = trace; return Ok(r2); }
                    Err(e) if is_no_api_access(&e) => {
                        self.disable_api_backend("OAuth token or API key has no direct API access");
                        // fall through to local model below
                    }
                    Err(e) if is_rate_limited(&e) && cfg.local.enabled => {
                        warn!("Anthropic rate-limited — falling back to local model");
                        match self.try_local(req, &shape, &pol, embedding.as_deref(), start, &mut trace).await {
                            Ok((Some(mut r), _)) => { r.trace = trace; return Ok(r); }
                            Ok((None, _)) => return Err(e),
                            Err(local_err) => { warn!("local model also failed: {local_err}"); return Err(e); }
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // ── Step 6: Budget gate ───────────────────────────────────────────
        // Only checked when budget tracking is enabled (API-key billing mode).
        // When disabled (subscription/Pro/Max), this gate is bypassed entirely.
        let budget_exceeded = if cfg.budget.enabled {
            self.budget.check().await?.is_exceeded()
        } else {
            false
        };
        if budget_exceeded {
            warn!("budget exceeded → local-only mode, API calls blocked");
        }

        // ── Step 7: Local model ───────────────────────────────────────────
        if cfg.local.enabled {
            match self.try_local(req, &shape, &pol, embedding.as_deref(), start, &mut trace).await {
                Ok((Some(mut routed), _)) => {
                    routed.trace = trace;
                    return Ok(routed);
                }
                Ok((None, _)) if budget_exceeded => {
                    return Err(ProxyError::BudgetExceeded(
                        "daily budget exceeded and local model confidence below floor".to_string()
                    ).into());
                }
                Ok((None, attempt)) => {
                    if self.api_blocked(&cfg) {
                        anyhow::bail!("local model confidence below floor and API backend is disabled");
                    }
                    debug!("local model confidence too low → api");
                    trace.miss_reason = Some("low_confidence".to_string());
                    let sj = scores_json(&score, novelty_t);
                    let api_result = self.call_api_and_cache(req, &shape, &pol, start,
                                                             Some("low_confidence"), attempt, Some(sj), embedding.as_deref()).await;
                    match api_result {
                        Ok(mut r2) => { r2.trace = trace; return Ok(r2); }
                        Err(e) if is_no_api_access(&e) => {
                            self.disable_api_backend("OAuth token or API key has no direct API access");
                            anyhow::bail!("API access denied and local model confidence below floor");
                        }
                        Err(e) if is_rate_limited(&e) => {
                            warn!("Anthropic rate-limited after low-confidence local attempt — returning rate limit error");
                            return Err(e);
                        }
                        Err(e) => return Err(e),
                    }
                }
                Err(e) if budget_exceeded => {
                    return Err(ProxyError::BudgetExceeded(
                        format!("daily budget exceeded and local model unavailable: {e}")
                    ).into());
                }
                Err(e) => warn!("local model error (escalating to api): {e}"),
            }
        } else if budget_exceeded {
            return Err(ProxyError::BudgetExceeded(
                "daily budget exceeded and local model disabled".to_string()
            ).into());
        }

        // ── Step 8: Anthropic API (last resort) ───────────────────────────
        // Only reachable when local model errored and budget is NOT exceeded.
        trace.miss_reason = Some("local_error".to_string());
        let sj = scores_json(&score, novelty_t);
        let mut r2 = self.call_api_and_cache(req, &shape, &pol, start, Some("local_error"), None, Some(sj), embedding.as_deref()).await?;
        r2.trace = trace;
        Ok(r2)
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    async fn try_local(
        &self,
        req:       &MessagesRequest,
        shape:     &domain::ShapeKey,
        pol:       &policy::CachePolicy,
        embedding: Option<&[f32]>,
        start:     Instant,
        trace:     &mut RouteTrace,
    ) -> Result<(Option<RoutedResponse>, Option<String>)> {
        let cfg = self.cfg.load();

        // ── Layer 2: domain knowledge document ───────────────────────────────
        // If a distilled knowledge document exists for this domain, prepend it
        // as a system prompt prefix so the local model has accumulated context.
        let sys_augmented;
        let req = if cfg.learning.enabled {
            if let Ok(Some(doc)) = self.cache.load_knowledge_doc(&shape.domain).await {
                debug!("learning: injecting domain knowledge doc ({} chars)", doc.len());
                trace.l2_doc       = true;
                trace.l2_doc_chars = doc.len();
                let prefix = format!("## {} Domain Knowledge\n{}", shape.domain, doc);
                sys_augmented = req.with_system_prefix(&prefix);
                &sys_augmented
            } else { req }
        } else { req };

        // ── Layer 1: few-shot context injection ───────────────────────────────
        // Pull the top-K semantically related Q&A pairs from the cache and
        // prepend them as prior conversation turns.  The local model then
        // answers with the benefit of how Claude handled similar prompts before.
        let augmented;
        let effective_req = if cfg.learning.enabled {
            if let Some(emb) = embedding {
                let examples = self.fewshot_examples(shape, emb, &cfg).await;
                // Populate trace: contrast pairs are marked with sim=0.0
                let contrast_count = examples.iter().filter(|e| e.sim == 0.0).count();
                let shot_count = examples.len().saturating_sub(contrast_count);
                let sims: Vec<f64> = examples.iter().filter(|e| e.sim > 0.0).map(|e| e.sim).collect();
                trace.l1_shots   = shot_count;
                trace.l5_contrast = contrast_count > 0;
                if !sims.is_empty() {
                    trace.l1_min_sim = sims.iter().cloned().fold(f64::INFINITY, f64::min);
                    trace.l1_max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                }
                if !examples.is_empty() {
                    debug!("learning: injecting {} few-shot examples (sim {:.2}–{:.2})",
                        examples.len(),
                        examples.last().map(|e| e.sim).unwrap_or(0.0),
                        examples.first().map(|e| e.sim).unwrap_or(0.0));
                    augmented = req.with_fewshot_context(&examples);
                    &augmented
                } else { req }
            } else { req }
        } else { req };

        let result = self.local.complete(effective_req).await?;
        let conf   = result.confidence.unwrap_or(0.0).clamp(0.0, 1.0);

        // Layer 6: apply per-(domain, intent) calibration bias to correct for
        // the local model's systematic over/under-confidence before gate check.
        let bias = {
            let cal = self.calibration.load();
            cal.get(&(shape.domain.clone(), shape.intent.clone())).copied().unwrap_or(0.0)
        };
        let calibrated_conf = (conf + bias).clamp(0.0, 1.0);
        trace.confidence = Some(calibrated_conf);

        if calibrated_conf < cfg.local.confidence_floor {
            // Layer 5: surface the attempt text so the caller can pair it with
            // the correct API answer as a contrast example for distillation.
            let attempt_text = result.response.text_content();
            let attempt = if attempt_text.is_empty() { None } else { Some(attempt_text) };
            return Ok((None, attempt));
        }

        if (bias).abs() > 0.01 {
            info!("local model hit (confidence={conf:.2} → calibrated={calibrated_conf:.2}, bias={bias:+.2})");
        } else {
            info!("local model hit (confidence={calibrated_conf:.2})");
        }
        let latency_ms = start.elapsed().as_millis() as u64;
        let saved      = estimate_api_cost(&self.budget, req).await;

        // Cache the local response for next time
        if pol.should_cache() {
            let resp_json = serde_json::to_string(&result.response)?;
            let cache_id  = self.cache.store(
                shape,
                &req.prompt_text(),
                req.normalized_system().as_deref(),
                &resp_json,
                "ollama",
                result.confidence,
                pol.ttl_secs,
                pol.shareable && cfg.federation.share_cache,
                false,
            ).await?;

            if cfg.embedding.enabled {
                if let Some(emb) = embedding {
                    let _ = self.cache.store_embedding(&cache_id, emb, self.embedder.model()).await;
                }
            }
        }

        let sj = Some(serde_json::json!({
            "novelty":     trace.novelty_score,
            "complexity":  trace.complexity_score,
            "consequence": trace.consequence_score,
            "threshold":   trace.l3_threshold,
        }).to_string());
        self.log(shape, RouteDecision::LocalModel, "ollama", latency_ms,
            Some(result.response.usage.input_tokens as i64), req, saved, None, sj,
            Some(result.response.usage.output_tokens as i64)).await;
        Ok((Some(RoutedResponse {
            response:   result.response,
            decision:   RouteDecision::LocalModel,
            latency_ms,
            saved_usd:  saved,
            trace:      RouteTrace::default(), // overwritten by route() after return
        }), None))
    }

    /// Build few-shot examples from cache entries that are semantically related
    /// but below the cache-serve threshold.  Returns an empty vec if none qualify.
    /// When `contrast_in_fewshot` is enabled, appends one labeled contrast pair
    /// so the local model sees a wrong/correct example for this domain.
    async fn fewshot_examples(
        &self,
        shape:     &domain::ShapeKey,
        embedding: &[f32],
        cfg:       &AppConfig,
    ) -> Vec<FewShotExample> {
        let min_sim   = cfg.learning.min_sim;
        let max_sim   = cfg.embedding.sim_threshold;
        let max_chars = cfg.learning.max_answer_chars;

        let mut examples: Vec<FewShotExample> = self.cache
            .lookup_fewshot(&shape.domain, embedding, min_sim, max_sim, cfg.learning.fewshot_k, self.embedder.model())
            .await
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(entry, sim)| {
                let full = extract_response_text(&entry.response)?;
                let answer = if full.len() > max_chars {
                    format!("{}…", truncate_chars(&full, max_chars))
                } else {
                    full
                };
                Some(FewShotExample { question: entry.prompt_text, answer, sim })
            })
            .collect();

        // Layer 5: optionally inject one contrast pair as a negative example so
        // the model learns what NOT to do, not just what good answers look like.
        if cfg.learning.contrast_in_fewshot {
            if let Ok(pairs) = self.cache.contrast_pairs_for_domain(&shape.domain, 1).await {
                if let Some(pair) = pairs.into_iter().next() {
                    if !pair.prompt_text.is_empty() {
                        let attempt = if pair.local_attempt.len() > max_chars {
                            format!("{}…", truncate_chars(&pair.local_attempt, max_chars))
                        } else { pair.local_attempt.clone() };
                        let correct = if pair.correct_answer.len() > max_chars {
                            format!("{}…", truncate_chars(&pair.correct_answer, max_chars))
                        } else { pair.correct_answer.clone() };
                        let contrast_answer = format!(
                            "[Previous incorrect attempt]:\n{attempt}\n\n\
                             [Correct answer]:\n{correct}"
                        );
                        examples.push(FewShotExample {
                            question: pair.prompt_text,
                            answer:   contrast_answer,
                            sim:      0.0,
                        });
                        debug!("learning: injected 1 contrast pair for domain '{}'", shape.domain);
                    }
                }
            }
        }

        examples
    }

    async fn call_api_and_cache(
        &self,
        req:           &MessagesRequest,
        shape:         &domain::ShapeKey,
        pol:           &policy::CachePolicy,
        start:         Instant,
        miss_reason:   Option<&str>,
        local_attempt: Option<String>,
        scores_json:   Option<String>,
        embedding:     Option<&[f32]>,
    ) -> Result<RoutedResponse> {
        let cfg = self.cfg.load();

        // Draft-verify: only fires when we're already committed to an API call.
        // Skipped for fast-path bypasses that never saw the routing gate.
        let skip_draft = matches!(miss_reason, Some("tool_use") | Some("policy_bypass") | Some("user_direct"));
        let draft_req_owned: Option<MessagesRequest> = if !skip_draft
            && cfg.routing.draft_verify_enabled
            && cfg.embedding.enabled
        {
            if let Some(emb) = embedding {
                let near = self.cache.lookup_fewshot(
                    &shape.domain, emb,
                    cfg.routing.draft_verify_min_sim,
                    cfg.embedding.sim_threshold,
                    1, self.embedder.model(),
                ).await.unwrap_or_default();
                near.into_iter().next().and_then(|(entry, sim)| {
                    extract_response_text(&entry.response).map(|draft| {
                        let sim_pct = (sim * 100.0).round() as u32;
                        info!("draft-verify: near-miss (sim={sim:.3}) → enriched API call");
                        req.with_draft_context(&draft, sim_pct)
                    })
                })
            } else { None }
        } else { None };

        let used_draft    = draft_req_owned.is_some();
        let effective_req = draft_req_owned.as_ref().unwrap_or(req);
        let log_reason    = if used_draft { Some("draft_verify") } else { miss_reason };

        let result     = self.api.complete(effective_req).await?;
        let latency_ms = start.elapsed().as_millis() as u64;
        self.record_spend(&result).await;

        if pol.should_cache() && !result.response.uses_tools() {
            let resp_json = serde_json::to_string(&result.response)?;
            // Always cache against the original prompt key (not the draft-enriched one)
            let cache_id  = self.cache.store(
                shape,
                &req.prompt_text(),
                req.normalized_system().as_deref(),
                &resp_json,
                "anthropic",
                None,
                pol.ttl_secs,
                pol.shareable && cfg.federation.share_cache,
                false,
            ).await?;

            if cfg.embedding.enabled {
                if let Some(emb) = embedding {
                    let _ = self.cache.store_embedding(&cache_id, emb, self.embedder.model()).await;
                }
            }

            // Layer 5: if the local model had a low-confidence attempt, record the
            // contrast pair now that we have the correct API answer.
            // Skip when draft-verify was used: the API responded to a draft-enriched
            // prompt, so its output is not a clean ground-truth answer to the original
            // question and would corrupt the contrast pair training signal.
            if cfg.learning.contrast_enabled && !used_draft {
                if let (Some(attempt), Some("low_confidence")) = (&local_attempt, miss_reason) {
                    let correct = result.response.text_content();
                    if !correct.is_empty() {
                        let _ = self.cache.store_escalation_pair(
                            Some(&cache_id),
                            &shape.domain,
                            &shape.intent,
                            &req.prompt_text(),
                            attempt,
                            &correct,
                            None,
                        ).await;
                        debug!("contrast pair stored for {}/{}", shape.domain, shape.intent);
                    }
                }
            }
        }

        self.log(shape, RouteDecision::Api, "anthropic", latency_ms,
            Some(result.response.usage.input_tokens as i64), req, 0.0, log_reason, scores_json,
            Some(result.response.usage.output_tokens as i64)).await;
        Ok(RoutedResponse {
            response:   result.response,
            decision:   RouteDecision::Api,
            latency_ms,
            saved_usd:  0.0,
            trace:      RouteTrace::default(), // overwritten by route() after return
        })
    }

    async fn record_spend(&self, result: &BackendResult) {
        if !self.cfg.load().budget.enabled { return; }
        let u = &result.response.usage;
        let _ = self.budget.record(
            &result.response.model,
            u.input_tokens,
            u.output_tokens,
        ).await;
    }

    async fn log(
        &self,
        shape:       &domain::ShapeKey,
        decision:    RouteDecision,
        backend:     &str,
        latency_ms:  u64,
        tokens_in:   Option<i64>,
        req:         &MessagesRequest,
        saved_usd:   f64,
        miss_reason: Option<&str>,
        scores_json: Option<String>,
        tokens_out:  Option<i64>,
    ) {
        let tin = tokens_in.unwrap_or_else(|| req.estimated_input_tokens() as i64);
        let _ = self.cache.log_routing(
            &shape.display(),
            &shape.domain,
            &shape.intent,
            decision.as_str(),
            backend,
            latency_ms as i64,
            Some(tin),
            tokens_out,
            if saved_usd > 0.0 { Some(saved_usd) } else { None },
            miss_reason,
            scores_json.as_deref(),
        ).await;
    }

    /// Cache an accumulated streaming response and record budget spend.
    /// Called fire-and-forget after stream_to_channel completes.
    /// `message_id` is the real Anthropic message ID from the SSE stream (may be empty on error).
    pub async fn cache_streamed(
        &self,
        req:          &MessagesRequest,
        text:         &str,
        message_id:   &str,
        input_tokens: u32,
        output_tokens: u32,
        stop_reason:  &str,
        embedding:    Option<Vec<f32>>,
    ) {
        use crate::backend::{ContentBlock, MessagesResponse, Usage};

        let cfg = self.cfg.load();

        // Record spend only when budget tracking is enabled.
        if cfg.budget.enabled {
            let _ = self.budget.record(&cfg.api.model, input_tokens, output_tokens).await;
        }

        if text.is_empty() {
            return;
        }

        let prompt = req.prompt_text();
        let shape  = domain::classify(&prompt);
        let pol    = policy::infer(&shape, &prompt, &cfg);

        if pol.bypass_cache || !pol.should_cache() {
            return;
        }

        let id = if message_id.is_empty() {
            format!("msg_{}", uuid::Uuid::new_v4().simple())
        } else {
            message_id.to_string()
        };

        let response = MessagesResponse {
            id,
            kind:        "message".to_string(),
            role:        "assistant".to_string(),
            content:     vec![ContentBlock { kind: "text".to_string(), text: Some(text.to_string()), extra: Default::default() }],
            model:       cfg.api.model.clone(),
            stop_reason: Some(stop_reason.to_string()),
            usage:       Usage { input_tokens, output_tokens },
        };

        let resp_json = match serde_json::to_string(&response) {
            Ok(j) => j,
            Err(e) => { warn!("stream cache serialize: {e}"); return; }
        };

        let shareable = pol.shareable && cfg.federation.share_cache;
        match self.cache.store(&shape, &prompt, req.normalized_system().as_deref(), &resp_json, "anthropic", None, pol.ttl_secs, shareable, false).await {
            Ok(cache_id) => {
                info!("stream cached: {} ({output_tokens} output tokens)", &cache_id[..8]);
                if cfg.embedding.enabled {
                    let emb = match embedding {
                        Some(v) => Some(v),
                        None    => self.embedder.embed(&prompt).await.ok(),
                    };
                    if let Some(emb) = emb {
                        let _ = self.cache.store_embedding(&cache_id, &emb, self.embedder.model()).await;
                    }
                }
            }
            Err(e) => warn!("stream cache store: {e}"),
        }
    }
}

fn is_rate_limited(e: &anyhow::Error) -> bool {
    if let Some(ProxyError::RateLimited(_)) = e.downcast_ref::<ProxyError>() {
        return true;
    }
    let s = e.to_string();
    // Matches both rate_limit_error (request/token quota) and overloaded_error
    // (capacity pressure) — both arrive as HTTP 429 and both warrant local fallback.
    s.contains("429") && (
        s.contains("rate_limit") ||
        s.contains("Too Many Requests") ||
        s.contains("overloaded")
    )
}

/// True when the API rejected us permanently — OAuth token on api.anthropic.com,
/// or API key with no billing.
fn is_no_api_access(e: &anyhow::Error) -> bool {
    if let Some(ProxyError::NoApiAccess(_)) = e.downcast_ref::<ProxyError>() {
        return true;
    }
    e.to_string().contains("no_api_access")
}

/// Truncate `s` to at most `max_bytes` bytes, aligned to a valid UTF-8 char boundary.
fn truncate_chars(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes { return s; }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) { end -= 1; }
    &s[..end]
}

async fn estimate_api_cost(budget: &BudgetLedger, req: &MessagesRequest) -> f64 {
    let pricing = budget.current_pricing().await;
    let tin     = req.estimated_input_tokens();
    let tout    = ((tin as f64 * 0.6) as u32).min(req.max_tokens);
    pricing.estimate_cost(tin, tout)
}

/// Deserialize a stored JSON response and extract its text content.
/// Returns None if the JSON is malformed or the response contains no text blocks.
fn extract_response_text(response_json: &str) -> Option<String> {
    let resp: MessagesResponse = serde_json::from_str(response_json).ok()?;
    let text = resp.text_content();
    if text.is_empty() { None } else { Some(text) }
}

/// Serialize routing gate scores to JSON for storage in `routing_log.scores_json`.
fn scores_json(score: &crate::scoring::RoutingScore, threshold: f64) -> String {
    serde_json::json!({
        "novelty":     score.novelty,
        "complexity":  score.complexity,
        "consequence": score.consequence,
        "threshold":   threshold,
    }).to_string()
}
