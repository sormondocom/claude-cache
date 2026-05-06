use crate::domain::{self, ShapeKey};

/// Three-axis routing score. All three axes must be below threshold for the
/// request to be served locally — any axis above threshold escalates to API.
#[derive(Debug, Clone)]
pub struct RoutingScore {
    pub novelty:     f64,
    pub complexity:  f64,
    pub consequence: f64,
}

impl RoutingScore {
    /// Returns true if ALL axes are below their thresholds → serve locally.
    pub fn should_use_local(
        &self,
        novelty_thresh:     f64,
        complexity_thresh:  f64,
        consequence_thresh: f64,
    ) -> bool {
        self.novelty     < novelty_thresh
            && self.complexity  < complexity_thresh
            && self.consequence < consequence_thresh
    }

    pub fn display(&self) -> String {
        format!(
            "nov={:.2} cplx={:.2} cons={:.2}",
            self.novelty, self.complexity, self.consequence
        )
    }
}

/// Score a prompt for routing. `cache_hit_count` is how many times we've seen
/// this shape before (0 = totally novel).
pub fn score_prompt(
    shape: &ShapeKey,
    prompt: &str,
    cache_hit_count: i64,
    semantic_similarity: Option<f64>,
) -> RoutingScore {
    let novelty     = novelty_score(cache_hit_count, semantic_similarity);
    let complexity  = shape.complexity;
    let consequence = domain::consequence_score(&shape.domain, &shape.intent);

    // Safety bump: review + high-complexity languages always get API treatment
    let consequence = if shape.intent == "review" && complexity > 0.5 {
        (consequence + 0.20).min(1.0)
    } else {
        consequence
    };

    // Long prompts with code blocks are more novel/complex
    let novelty = if prompt.contains("```") && cache_hit_count == 0 {
        (novelty + 0.10).min(1.0)
    } else {
        novelty
    };

    RoutingScore { novelty, complexity, consequence }
}

fn novelty_score(hit_count: i64, semantic_sim: Option<f64>) -> f64 {
    // Base novelty from cache history
    let base = match hit_count {
        0 => 0.80,
        1 => 0.50,
        2..=4 => 0.35,
        5..=19 => 0.20,
        _ => 0.05,
    };

    // If we have a semantic match, reduce novelty proportionally
    if let Some(sim) = semantic_sim {
        // sim=1.0 → novelty=0.0, sim=0.88 → novelty still relatively low
        let sem_reduction = sim * 0.70;
        return (base - sem_reduction).max(0.0);
    }

    base
}
