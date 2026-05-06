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

pub struct Router {
    cfg:      Arc<AppConfig>,
    cache:    Arc<CacheStore>,
    budget:   Arc<BudgetLedger>,
    embedder: Arc<dyn Embedder>,
    local:    Arc<dyn ModelBackend>,
    api:      Arc<dyn ModelBackend>,
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
        Router { cfg, cache, budget, embedder, local, api }
    }

    pub async fn route(&self, req: &MessagesRequest) -> Result<RoutedResponse> {
        let start   = Instant::now();
        let prompt  = req.prompt_text();

        // ── Step 0: tool-use fast-path ────────────────────────────────────
        // Tools require live API semantics; never route locally.
        if req.has_tools() {
            debug!("tool-use fast-path → api");
            return self.call_api(req, start).await;
        }

        // ── Step 1: Classify ─────────────────────────────────────────────
        let shape = domain::classify(&prompt);
        debug!("classified: {}", shape.display());

        // ── Step 2: Policy ───────────────────────────────────────────────
        let pol = policy::infer(&shape, &prompt, &self.cfg);
        if pol.bypass_cache {
            debug!("policy bypass → api");
            return self.call_api(req, start).await;
        }

        // ── Step 3: Exact cache ──────────────────────────────────────────
        if let Some(entry) = self.cache.lookup_exact(&prompt).await? {
            info!("exact cache hit: {}", &entry.id[..8]);
            let resp = serde_json::from_str(&entry.response)?;
            let latency_ms = start.elapsed().as_millis() as u64;
            let saved = estimate_api_cost(&self.budget, req).await;
            self.log(&shape, RouteDecision::ExactCache, "cache", latency_ms, req, saved).await;
            return Ok(RoutedResponse {
                response: resp,
                decision: RouteDecision::ExactCache,
                latency_ms,
                saved_usd: saved,
            });
        }

        // ── Step 4: Semantic cache ────────────────────────────────────────
        let semantic_sim = if self.cfg.embedding.enabled {
            match self.embedder.embed(&prompt).await {
                Ok(emb) => {
                    let hits = self.cache.lookup_semantic(
                        &shape.domain,
                        &emb,
                        self.cfg.embedding.sim_threshold,
                        1,
                    ).await?;
                    if let Some((entry, sim)) = hits.into_iter().next() {
                        info!("semantic cache hit (sim={sim:.3}): {}", &entry.id[..8]);
                        let resp = serde_json::from_str(&entry.response)?;
                        let latency_ms = start.elapsed().as_millis() as u64;
                        let saved = estimate_api_cost(&self.budget, req).await;
                        self.log(&shape, RouteDecision::SemanticCache, "cache", latency_ms, req, saved).await;
                        return Ok(RoutedResponse {
                            response: resp,
                            decision: RouteDecision::SemanticCache,
                            latency_ms,
                            saved_usd: saved,
                        });
                    }
                    None
                }
                Err(e) => {
                    warn!("embedding failed, skipping semantic cache: {e}");
                    None
                }
            }
        } else {
            None
        };

        // ── Step 5: Routing gate ──────────────────────────────────────────
        // How many times have we seen this shape before?
        let hit_count = {
            // Re-use the exact-miss result — 0 since step 3 found nothing
            0i64
        };
        let score = scoring::score_prompt(&shape, &prompt, hit_count, semantic_sim);
        debug!("routing score: {}", score.display());

        let r = &self.cfg.routing;
        if !score.should_use_local(r.novelty_threshold, r.complexity_threshold, r.consequence_threshold) {
            debug!("routing gate → api ({})", score.display());
            return self.call_api_and_cache(req, &shape, &pol, start).await;
        }

        // ── Step 6: Budget gate ───────────────────────────────────────────
        match self.budget.check().await? {
            crate::budget::BudgetStatus::Exceeded { spent_usd, limit_usd } => {
                warn!("budget exceeded (${spent_usd:.4} / ${limit_usd:.4}) → force local");
                // fall through to local
            }
            _ => {}
        }

        // ── Step 7: Local model ───────────────────────────────────────────
        if self.cfg.local.enabled {
            match self.try_local(req, &shape, &pol, start).await {
                Ok(Some(routed)) => return Ok(routed),
                Ok(None)         => debug!("local model confidence too low → api"),
                Err(e)           => warn!("local model error (escalating to api): {e}"),
            }
        }

        // ── Step 8: Anthropic API (last resort) ───────────────────────────
        self.call_api_and_cache(req, &shape, &pol, start).await
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
                &resp_json,
                "ollama",
                result.confidence,
                pol.ttl_secs,
                pol.shareable,
            ).await?;

            // Store embedding if enabled
            if self.cfg.embedding.enabled {
                if let Ok(emb) = self.embedder.embed(&req.prompt_text()).await {
                    let _ = self.cache.store_embedding(&cache_id, &emb, self.embedder.model()).await;
                }
            }
        }

        self.log(shape, RouteDecision::LocalModel, "ollama", latency_ms, req, saved).await;
        Ok(Some(RoutedResponse {
            response:   result.response,
            decision:   RouteDecision::LocalModel,
            latency_ms,
            saved_usd:  saved,
        }))
    }

    async fn call_api(&self, req: &MessagesRequest, start: Instant) -> Result<RoutedResponse> {
        let result     = self.api.complete(req).await?;
        let latency_ms = start.elapsed().as_millis() as u64;
        self.record_spend(&result, req).await;
        Ok(RoutedResponse {
            response:   result.response,
            decision:   RouteDecision::Api,
            latency_ms,
            saved_usd:  0.0,
        })
    }

    async fn call_api_and_cache(
        &self,
        req:   &MessagesRequest,
        shape: &domain::ShapeKey,
        pol:   &policy::CachePolicy,
        start: Instant,
    ) -> Result<RoutedResponse> {
        let result     = self.api.complete(req).await?;
        let latency_ms = start.elapsed().as_millis() as u64;
        self.record_spend(&result, req).await;

        if pol.should_cache() {
            let resp_json = serde_json::to_string(&result.response)?;
            let cache_id  = self.cache.store(
                shape,
                &req.prompt_text(),
                &resp_json,
                "anthropic",
                None,
                pol.ttl_secs,
                pol.shareable,
            ).await?;

            if self.cfg.embedding.enabled {
                if let Ok(emb) = self.embedder.embed(&req.prompt_text()).await {
                    let _ = self.cache.store_embedding(&cache_id, &emb, self.embedder.model()).await;
                }
            }
        }

        self.log(shape, RouteDecision::Api, "anthropic", latency_ms, req, 0.0).await;
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
        shape:      &domain::ShapeKey,
        decision:   RouteDecision,
        backend:    &str,
        latency_ms: u64,
        req:        &MessagesRequest,
        saved_usd:  f64,
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
        ).await;
    }
}

async fn estimate_api_cost(budget: &BudgetLedger, req: &MessagesRequest) -> f64 {
    let pricing = budget.current_pricing().await;
    let tin     = req.estimated_input_tokens();
    let tout    = (tin as f64 * 0.6) as u32;
    pricing.estimate_cost(tin, tout)
}
