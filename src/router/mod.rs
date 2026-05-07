use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::backend::{BackendResult, MessagesRequest, MessagesResponse, ModelBackend};
use crate::budget::BudgetLedger;
use crate::cache::CacheStore;
use crate::config::AppConfig;
use crate::domain;
use crate::embedding::Embedder;
use crate::federation::FederationClient;
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
}

#[derive(Clone)]
pub struct Router {
    cfg:        Arc<AppConfig>,
    cache:      Arc<CacheStore>,
    budget:     Arc<BudgetLedger>,
    embedder:   Arc<dyn Embedder>,
    local:      Arc<dyn ModelBackend>,
    api:        Arc<dyn ModelBackend>,
    federation: Option<Arc<FederationClient>>,
}

impl Router {
    pub fn new(
        cfg:      Arc<AppConfig>,
        cache:    Arc<CacheStore>,
        budget:   Arc<BudgetLedger>,
        embedder: Arc<dyn Embedder>,
        local:    Arc<dyn ModelBackend>,
        api:      Arc<dyn ModelBackend>,
    ) -> Self {
        Router { cfg, cache, budget, embedder, local, api, federation: None }
    }

    /// Attach a federation client.  Call after `new()` before first use.
    pub fn with_federation(mut self, fed: Arc<FederationClient>) -> Self {
        self.federation = Some(fed);
        self
    }

    pub async fn route(&self, req: &MessagesRequest) -> Result<RoutedResponse> {
        let start   = Instant::now();
        let prompt  = req.prompt_text();

        // ── Step 0: tool-use fast-path ────────────────────────────────────
        // Tools require live API semantics; never route locally.
        if req.has_tools() {
            debug!("tool-use fast-path → api");
            return self.call_api_and_cache(req, &domain::classify(&prompt),
                &policy::infer(&domain::classify(&prompt), &prompt, &self.cfg),
                start, Some("tool_use")).await;
        }

        // ── Step 1: Classify ─────────────────────────────────────────────
        let shape = domain::classify(&prompt);
        debug!("classified: {}", shape.display());

        // ── Step 2: Policy ───────────────────────────────────────────────
        let pol = policy::infer(&shape, &prompt, &self.cfg);
        if pol.bypass_cache {
            debug!("policy bypass → api");
            return self.call_api_and_cache(req, &shape, &pol, start, Some("policy_bypass")).await;
        }

        // ── Step 3: Exact cache ──────────────────────────────────────────
        if let Some(entry) = self.cache.lookup_exact(&prompt, req.normalized_system().as_deref()).await? {
            info!("exact cache hit: {}", &entry.id[..8]);
            let resp = serde_json::from_str(&entry.response)?;
            let latency_ms = start.elapsed().as_millis() as u64;
            let saved = estimate_api_cost(&self.budget, req).await;
            self.log(&shape, RouteDecision::ExactCache, "cache", latency_ms, req, saved, None).await;
            return Ok(RoutedResponse {
                response: resp,
                decision: RouteDecision::ExactCache,
                latency_ms,
                saved_usd: saved,
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
                let pol = policy::infer(&shape, &prompt, &self.cfg);
                if pol.should_cache() {
                    let _ = self.cache.store(&shape, &prompt, req.normalized_system().as_deref(),
                        &fed_entry.response, "federated", None, pol.ttl_secs, false, false).await;
                }
                let node_id = fed_entry.node_id.clone();
                self.log(&shape, RouteDecision::FederationPeer(node_id.clone()), "federation", latency_ms, req, saved, None).await;
                return Ok(RoutedResponse {
                    response: resp,
                    decision: RouteDecision::FederationPeer(node_id),
                    latency_ms,
                    saved_usd: saved,
                });
            }
        }

        // ── Step 4: Embedding (computed once, used for local + federation) ─
        let embedding = if self.cfg.embedding.enabled {
            match self.embedder.embed(&prompt).await {
                Ok(emb) => Some(emb),
                Err(e)  => { warn!("embedding failed, skipping semantic lookups: {e}"); None }
            }
        } else {
            None
        };

        // ── Step 4a: Local semantic cache ─────────────────────────────────
        // Also capture the best similarity found below threshold so the routing
        // gate can tell "near miss" apart from "truly never seen".
        let mut best_semantic_sim: Option<f64> = None;
        if let Some(ref emb) = embedding {
            let hits = self.cache.lookup_semantic(
                &shape.domain, emb, self.cfg.embedding.sim_threshold, 1,
            ).await?;
            if let Some((entry, sim)) = hits.into_iter().next() {
                info!("semantic cache hit (sim={sim:.3}): {}", &entry.id[..8]);
                let resp = serde_json::from_str(&entry.response)?;
                let latency_ms = start.elapsed().as_millis() as u64;
                let saved = estimate_api_cost(&self.budget, req).await;
                self.log(&shape, RouteDecision::SemanticCache, "cache", latency_ms, req, saved, None).await;
                return Ok(RoutedResponse {
                    response: resp,
                    decision: RouteDecision::SemanticCache,
                    latency_ms,
                    saved_usd: saved,
                });
            }
            // No hit above threshold — probe for near-miss similarity so the
            // routing gate knows how familiar this prompt shape actually is.
            best_semantic_sim = self.cache.best_semantic_sim(&shape.domain, emb).await.ok().flatten();
        }

        // ── Step 4b: Federation semantic lookup ───────────────────────────
        // Ask trusted peers for semantically similar entries using our embedding.
        if let (Some(ref fed), Some(ref emb)) = (&self.federation, &embedding) {
            if let Some((fed_entry, sim)) = fed.lookup_semantic(
                emb, &shape.domain, self.cfg.embedding.sim_threshold, 1,
            ).await {
                info!("federation semantic hit (sim={sim:.3}) from {}",
                    &fed_entry.node_id[..16.min(fed_entry.node_id.len())]);
                let resp: MessagesResponse = serde_json::from_str(&fed_entry.response)?;
                let latency_ms = start.elapsed().as_millis() as u64;
                let saved = estimate_api_cost(&self.budget, req).await;
                // Cache locally so future identical prompts skip the peer call.
                let pol = policy::infer(&shape, &prompt, &self.cfg);
                if pol.should_cache() {
                    if let Ok(cid) = self.cache.store(&shape, &prompt, req.normalized_system().as_deref(),
                            &fed_entry.response, "federated", None, pol.ttl_secs, false, false).await {
                        if let Some(ref emb2) = embedding {
                            let _ = self.cache.store_embedding(&cid, emb2, self.embedder.model()).await;
                        }
                    }
                }
                let node_id = fed_entry.node_id.clone();
                self.log(&shape, RouteDecision::FederationPeer(node_id.clone()), "federation", latency_ms, req, saved, None).await;
                return Ok(RoutedResponse {
                    response: resp,
                    decision: RouteDecision::FederationPeer(node_id),
                    latency_ms,
                    saved_usd: saved,
                });
            }
        }

        // ── Step 5: Routing gate ──────────────────────────────────────────
        // How familiar are we with this domain+intent shape?
        let hit_count = self.cache.domain_hit_count(&shape.domain, &shape.intent)
            .await
            .unwrap_or(0);
        let score = scoring::score_prompt(&shape, &prompt, hit_count, best_semantic_sim);
        debug!("routing score: {}", score.display());

        let r = &self.cfg.routing;
        if !score.should_use_local(r.novelty_threshold, r.complexity_threshold, r.consequence_threshold) {
            let miss = score.gate_miss_reason(r.novelty_threshold, r.complexity_threshold, r.consequence_threshold);
            debug!("routing gate → api ({}) miss={miss}", score.display());
            return self.call_api_and_cache(req, &shape, &pol, start, Some(miss)).await;
        }

        // ── Step 6: Budget gate ───────────────────────────────────────────
        let budget_exceeded = self.budget.check().await?.is_exceeded();
        if budget_exceeded {
            warn!("budget exceeded → local-only mode, API calls blocked");
        }

        // ── Step 7: Local model ───────────────────────────────────────────
        if self.cfg.local.enabled {
            match self.try_local(req, &shape, &pol, start).await {
                Ok(Some(routed)) => return Ok(routed),
                Ok(None) if budget_exceeded => {
                    anyhow::bail!("daily budget exceeded and local model confidence below floor");
                }
                Ok(None) => {
                    debug!("local model confidence too low → api");
                    return self.call_api_and_cache(req, &shape, &pol, start, Some("low_confidence")).await;
                }
                Err(e) if budget_exceeded => {
                    anyhow::bail!("daily budget exceeded and local model unavailable: {e}");
                }
                Err(e) => warn!("local model error (escalating to api): {e}"),
            }
        } else if budget_exceeded {
            anyhow::bail!("daily budget exceeded and local model disabled");
        }

        // ── Step 8: Anthropic API (last resort) ───────────────────────────
        // Only reachable when local model errored and budget is NOT exceeded.
        self.call_api_and_cache(req, &shape, &pol, start, Some("local_error")).await
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    async fn try_local(
        &self,
        req:   &MessagesRequest,
        shape: &domain::ShapeKey,
        pol:   &policy::CachePolicy,
        start: Instant,
    ) -> Result<Option<RoutedResponse>> {
        let result = self.local.complete(req).await?;
        let conf   = result.confidence.unwrap_or(0.0);

        if conf < self.cfg.local.confidence_floor {
            return Ok(None);
        }

        info!("local model hit (confidence={conf:.2})");
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
                pol.shareable && self.cfg.federation.share_cache,
                false,
            ).await?;

            // Store embedding if enabled
            if self.cfg.embedding.enabled {
                if let Ok(emb) = self.embedder.embed(&req.prompt_text()).await {
                    let _ = self.cache.store_embedding(&cache_id, &emb, self.embedder.model()).await;
                }
            }
        }

        self.log(shape, RouteDecision::LocalModel, "ollama", latency_ms, req, saved, None).await;
        Ok(Some(RoutedResponse {
            response:   result.response,
            decision:   RouteDecision::LocalModel,
            latency_ms,
            saved_usd:  saved,
        }))
    }



    async fn call_api_and_cache(
        &self,
        req:         &MessagesRequest,
        shape:       &domain::ShapeKey,
        pol:         &policy::CachePolicy,
        start:       Instant,
        miss_reason: Option<&str>,
    ) -> Result<RoutedResponse> {
        let result     = self.api.complete(req).await?;
        let latency_ms = start.elapsed().as_millis() as u64;
        self.record_spend(&result, req).await;

        if pol.should_cache() {
            let resp_json = serde_json::to_string(&result.response)?;
            let cache_id  = self.cache.store(
                shape,
                &req.prompt_text(),
                req.normalized_system().as_deref(),
                &resp_json,
                "anthropic",
                None,
                pol.ttl_secs,
                pol.shareable && self.cfg.federation.share_cache,
                false,
            ).await?;

            if self.cfg.embedding.enabled {
                if let Ok(emb) = self.embedder.embed(&req.prompt_text()).await {
                    let _ = self.cache.store_embedding(&cache_id, &emb, self.embedder.model()).await;
                }
            }
        }

        self.log(shape, RouteDecision::Api, "anthropic", latency_ms, req, 0.0, miss_reason).await;
        Ok(RoutedResponse {
            response:   result.response,
            decision:   RouteDecision::Api,
            latency_ms,
            saved_usd:  0.0,
        })
    }

    async fn record_spend(&self, result: &BackendResult, req: &MessagesRequest) {
        let u = &result.response.usage;
        let _ = self.budget.record(
            &result.response.model,
            u.input_tokens.max(req.estimated_input_tokens()),
            u.output_tokens,
        ).await;
    }

    async fn log(
        &self,
        shape:       &domain::ShapeKey,
        decision:    RouteDecision,
        backend:     &str,
        latency_ms:  u64,
        req:         &MessagesRequest,
        saved_usd:   f64,
        miss_reason: Option<&str>,
    ) {
        let tin = req.estimated_input_tokens() as i64;
        let _ = self.cache.log_routing(
            &shape.display(),
            decision.as_str(),
            backend,
            latency_ms as i64,
            Some(tin),
            None,
            if saved_usd > 0.0 { Some(saved_usd) } else { None },
            miss_reason,
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
    ) {
        use crate::backend::{ContentBlock, MessagesResponse, Usage};

        // Always record spend — streaming bypasses the sync budget path entirely.
        let _ = self.budget.record(&self.cfg.api.model, input_tokens, output_tokens).await;

        if req.has_tools() || text.is_empty() {
            return;
        }

        let prompt = req.prompt_text();
        let shape  = domain::classify(&prompt);
        let pol    = policy::infer(&shape, &prompt, &self.cfg);

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
            content:     vec![ContentBlock { kind: "text".to_string(), text: Some(text.to_string()) }],
            model:       self.cfg.api.model.clone(),
            stop_reason: Some("end_turn".to_string()),
            usage:       Usage { input_tokens, output_tokens },
        };

        let resp_json = match serde_json::to_string(&response) {
            Ok(j) => j,
            Err(e) => { warn!("stream cache serialize: {e}"); return; }
        };

        let shareable = pol.shareable && self.cfg.federation.share_cache;
        match self.cache.store(&shape, &prompt, req.normalized_system().as_deref(), &resp_json, "anthropic", None, pol.ttl_secs, shareable, false).await {
            Ok(cache_id) => {
                info!("stream cached: {} ({output_tokens} output tokens)", &cache_id[..8]);
                if self.cfg.embedding.enabled {
                    if let Ok(emb) = self.embedder.embed(&prompt).await {
                        let _ = self.cache.store_embedding(&cache_id, &emb, self.embedder.model()).await;
                    }
                }
            }
            Err(e) => warn!("stream cache store: {e}"),
        }
    }
}

async fn estimate_api_cost(budget: &BudgetLedger, req: &MessagesRequest) -> f64 {
    let pricing = budget.current_pricing().await;
    let tin     = req.estimated_input_tokens();
    let tout    = (tin as f64 * 0.6) as u32;
    pricing.estimate_cost(tin, tout)
}
