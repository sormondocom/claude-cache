use anyhow::Result;
use chrono::Utc;
use sha2::{Digest, Sha256};
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions},
    Row,
};
use std::str::FromStr;

use crate::domain::{self, ShapeKey};

// ── Types ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// SHA256(normalized_system + normalized_prompt) — the DHT key for federation
    pub id:          String,
    pub domain:      String,
    pub intent:      String,
    pub complexity:  f64,
    pub prompt_text: String,
    pub response:    String,
    pub model_used:  String,
    pub confidence:  Option<f64>,
    pub created_at:  i64,
    pub expires_at:  Option<i64>,
    pub hit_count:   i64,
    pub last_hit_at: Option<i64>,
    pub node_id:     String,
    pub shared:      bool,
    /// Pinned entries are never evicted by TTL expiry or size-limit LRU.
    pub pinned:      bool,
}

#[derive(Debug, Clone)]
pub struct CacheEmbedding {
    pub cache_id:  String,
    pub embedding: Vec<f32>,
    pub model:     String,
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries:  i64,
    pub total_hits:     i64,
    pub shared_entries: i64,
    pub db_size_bytes:  i64,
}

/// Per-(domain, intent) escalation statistics used by the threshold adaptor.
#[derive(Debug, Clone)]
pub struct EscalationStat {
    pub domain:          String,
    pub intent:          String,
    pub escalation_rate: f64,
    pub sample_count:    i64,
}

/// Lightweight projection used by the browse/search endpoints.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheEntrySummary {
    pub id:             String,
    pub domain:         String,
    pub intent:         String,
    /// First 200 characters of the prompt.
    pub prompt_preview: String,
    pub model_used:     String,
    pub hit_count:      i64,
    pub pinned:         bool,
    pub created_at:     i64,
    pub expires_at:     Option<i64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RoutingLogEntry {
    pub shape_key:   String,
    pub domain:      Option<String>,
    pub intent:      Option<String>,
    pub decision:    String,
    pub backend:     String,
    pub latency_ms:  i64,
    pub tokens_in:   Option<i64>,
    pub tokens_out:  Option<i64>,
    pub saved_usd:   Option<f64>,
    pub miss_reason: Option<String>,
    pub created_at:  i64,
}

/// Full entry returned by the export endpoint — includes the raw response payload.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheExportEntry {
    pub id:          String,
    pub domain:      String,
    pub intent:      String,
    pub prompt_text: String,
    pub response:    String,
    pub model_used:  String,
    pub hit_count:   i64,
    pub pinned:      bool,
    pub created_at:  i64,
    pub expires_at:  Option<i64>,
}

/// Lightweight summary of a stored domain knowledge document.
#[derive(Debug, Clone, serde::Serialize)]
pub struct KnowledgeDocSummary {
    pub domain:      String,
    pub entry_count: i64,
    pub version:     i64,
    pub updated_at:  i64,
}

/// Per-(domain, intent) quality signal counts from `response_feedback`.
/// Used by `ThresholdAdaptor::adapt()` to blend explicit user feedback into
/// the escalation rate so routing thresholds reflect satisfaction, not just
/// implicit routing failures.
#[derive(Debug, Clone)]
pub struct QualityStat {
    pub domain:     String,
    pub intent:     String,
    pub bad_count:  i64,
    pub good_count: i64,
}

/// A (wrong attempt, correct answer) pair captured when the local model tried
/// but its confidence was too low.  Used by the distiller (Layer 2 + 5) to
/// synthesize "avoid this" knowledge into domain documents, and optionally by
/// few-shot injection as a labeled negative example.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContrastPair {
    pub id:               String,
    pub domain:           String,
    pub intent:           String,
    pub prompt_text:      String,
    pub local_attempt:    String,
    pub correct_answer:   String,
    pub local_confidence: Option<f64>,
    pub created_at:       i64,
}

// ── Store ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct CacheStore {
    pool:    SqlitePool,
    node_id: String,
    db_path: String,
}

impl CacheStore {
    pub async fn open(db_path: &str, node_id: &str) -> Result<Self> {
        let opts = SqliteConnectOptions::from_str(&format!("sqlite:{db_path}"))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(opts)
            .await?;

        let store = CacheStore { pool, node_id: node_id.to_string(), db_path: db_path.to_string() };
        store.migrate().await?;
        Ok(store)
    }

    async fn migrate(&self) -> Result<()> {
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS cache_entries (
                id           TEXT    PRIMARY KEY,
                domain       TEXT    NOT NULL,
                intent       TEXT    NOT NULL,
                complexity   REAL    NOT NULL,
                prompt_text  TEXT    NOT NULL,
                response     TEXT    NOT NULL,
                model_used   TEXT    NOT NULL,
                confidence   REAL,
                created_at   INTEGER NOT NULL,
                expires_at   INTEGER,
                hit_count    INTEGER NOT NULL DEFAULT 0,
                last_hit_at  INTEGER,
                node_id      TEXT    NOT NULL,
                shared       INTEGER NOT NULL DEFAULT 0,
                pinned       INTEGER NOT NULL DEFAULT 0
            )
        "#).execute(&self.pool).await?;

        // Additive migration — safe to run on an existing table; SQLite
        // returns an error if the column already exists which we ignore.
        let _ = sqlx::query(
            "ALTER TABLE cache_entries ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
        ).execute(&self.pool).await;

        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS cache_embeddings (
                cache_id   TEXT    PRIMARY KEY REFERENCES cache_entries(id) ON DELETE CASCADE,
                embedding  BLOB    NOT NULL,
                model      TEXT    NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS routing_log (
                id          TEXT    PRIMARY KEY,
                shape_key   TEXT    NOT NULL,
                domain      TEXT,
                intent      TEXT,
                decision    TEXT    NOT NULL,
                backend     TEXT    NOT NULL,
                latency_ms  INTEGER NOT NULL,
                tokens_in   INTEGER,
                tokens_out  INTEGER,
                saved_usd   REAL,
                created_at  INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_cache_domain ON cache_entries(domain)")
            .execute(&self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)")
            .execute(&self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_cache_shared ON cache_entries(shared)")
            .execute(&self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_cache_hits ON cache_entries(hit_count DESC)")
            .execute(&self.pool).await?;

        // Additive migrations — errors silently ignored if column already exists.
        let _ = sqlx::query(
            "ALTER TABLE cache_entries ADD COLUMN last_hit_at INTEGER"
        ).execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN miss_reason TEXT")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN domain TEXT")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN intent TEXT")
            .execute(&self.pool).await;
        // Task 14: persist real gate scores with each routing decision.
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN scores_json TEXT")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN tokens_in INTEGER")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN tokens_out INTEGER")
            .execute(&self.pool).await;

        // Task 9: FTS5 full-text search over prompt_text.
        // Errors silently ignored — falls back to LIKE search if FTS5 unavailable.
        let _ = sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS cache_entries_fts USING fts5(id UNINDEXED, prompt_text)"
        ).execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS cache_fts_insert AFTER INSERT ON cache_entries BEGIN \
             INSERT INTO cache_entries_fts(id, prompt_text) VALUES(new.id, new.prompt_text); END"
        ).execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS cache_fts_update AFTER UPDATE ON cache_entries BEGIN \
             DELETE FROM cache_entries_fts WHERE id = old.id; \
             INSERT INTO cache_entries_fts(id, prompt_text) VALUES(new.id, new.prompt_text); END"
        ).execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS cache_fts_delete AFTER DELETE ON cache_entries BEGIN \
             DELETE FROM cache_entries_fts WHERE id = old.id; END"
        ).execute(&self.pool).await;
        // Backfill FTS for entries that predate this migration (runs only when table is empty).
        if let Ok(row) = sqlx::query("SELECT COUNT(*) AS n FROM cache_entries_fts")
            .fetch_one(&self.pool).await {
            let n: i64 = row.get("n");
            if n == 0 {
                let _ = sqlx::query(
                    "INSERT INTO cache_entries_fts(id, prompt_text) SELECT id, prompt_text FROM cache_entries"
                ).execute(&self.pool).await;
            }
        }

        // Layer 2: distilled domain knowledge documents.
        // Each row is the synthesized reference doc for one domain, versioned so
        // the management portal can show how many distillation passes have run.
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS domain_knowledge (
                domain      TEXT    PRIMARY KEY,
                content     TEXT    NOT NULL,
                entry_count INTEGER NOT NULL DEFAULT 0,
                version     INTEGER NOT NULL DEFAULT 1,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        // Layer 3: per-(domain, intent) adaptive novelty threshold overrides.
        // Written by ThresholdAdaptor; read by the router on every request.
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS routing_thresholds (
                domain           TEXT NOT NULL,
                intent           TEXT NOT NULL,
                novelty_override REAL NOT NULL,
                escalation_rate  REAL NOT NULL,
                sample_count     INTEGER NOT NULL,
                computed_at      INTEGER NOT NULL,
                PRIMARY KEY (domain, intent)
            )
        "#).execute(&self.pool).await?;

        // Layer 4: explicit quality feedback from user annotations.
        // `![good]` / `![bad]` in prompts are stripped before forwarding and
        // recorded here so the adaptive threshold adaptor has a user-satisfaction
        // signal independent of implicit routing escalation rates.
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS response_feedback (
                id         TEXT    PRIMARY KEY,
                domain     TEXT    NOT NULL,
                intent     TEXT    NOT NULL,
                signal     TEXT    NOT NULL CHECK(signal IN ('good', 'bad', 'repeat')),
                source     TEXT    NOT NULL CHECK(source IN ('explicit', 'implicit')),
                created_at INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_feedback_domain \
             ON response_feedback(domain, created_at)"
        ).execute(&self.pool).await?;

        // Layer 5: contrastive failure learning.
        // When the local model attempts a prompt but confidence is too low to
        // serve the answer, both the attempt and the correct API response are
        // stored here.  Distillation (Layer 2) and few-shot injection (Layer 1)
        // use these pairs to teach the local model what to avoid.
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS escalation_pairs (
                id               TEXT    PRIMARY KEY,
                cache_id         TEXT,       -- FK to the winning API cache entry
                prompt_text      TEXT NOT NULL DEFAULT '',
                local_attempt    TEXT NOT NULL,
                correct_answer   TEXT NOT NULL,
                domain           TEXT NOT NULL,
                intent           TEXT NOT NULL,
                local_confidence REAL,
                created_at       INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        // Additive: add prompt_text to pre-existing DBs that were created without it.
        let _ = sqlx::query(
            "ALTER TABLE escalation_pairs ADD COLUMN prompt_text TEXT NOT NULL DEFAULT ''"
        ).execute(&self.pool).await;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_ep_domain \
             ON escalation_pairs(domain, created_at)"
        ).execute(&self.pool).await?;

        // Layer 6: confidence calibration log.
        // Stores (claimed_confidence, actual_similarity) pairs collected by the
        // CalibrationRunner so the router can correct for local model over/under-confidence.
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS calibration_log (
                id           TEXT    PRIMARY KEY,
                domain       TEXT    NOT NULL,
                intent       TEXT    NOT NULL,
                claimed_conf REAL    NOT NULL,
                actual_sim   REAL    NOT NULL,
                created_at   INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_cal_domain \
             ON calibration_log(domain, intent, created_at)"
        ).execute(&self.pool).await?;

        Ok(())
    }

    // ── Key derivation ───────────────────────────────────────────────────

    /// Content-addressed key: SHA256(optional_system_prefix + normalized_prompt).
    ///
    /// When `system` is provided, a SHA256 of the trimmed system text is mixed in
    /// so that requests with different system prompts never share a cache entry.
    /// Requests without a system prompt are unaffected — their keys are identical
    /// to the pre-system-prompt format.
    pub fn content_key(prompt: &str, system: Option<&str>) -> String {
        let normalized = normalize_prompt(prompt);
        let mut hasher = Sha256::new();
        if let Some(sys) = system.filter(|s| !s.trim().is_empty()) {
            // Mix in SHA256(system) so long system prompts don't bloat the key
            let sys_hash = Sha256::digest(sys.trim().as_bytes());
            hasher.update(b"sys:");
            hasher.update(sys_hash.as_slice());
            hasher.update(b"|");
        }
        hasher.update(normalized.as_bytes());
        hex::encode(hasher.finalize())
    }

    // ── Lookup ───────────────────────────────────────────────────────────

    /// Exact-match cache lookup by SHA256 key (includes system prompt).
    pub async fn lookup_exact(
        &self,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<Option<CacheEntry>> {
        let key = Self::content_key(prompt, system);
        let now = Utc::now().timestamp();

        let row = sqlx::query(
            "SELECT id, domain, intent, complexity, prompt_text, response, model_used,
                    confidence, created_at, expires_at, hit_count, last_hit_at, node_id, shared, pinned
             FROM cache_entries
             WHERE id = ? AND (expires_at IS NULL OR expires_at > ?)"
        )
        .bind(&key)
        .bind(now)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let entry = row_to_entry(row);
            // Validate the response is parseable before bumping the hit count.
            // A corrupted row that can't be deserialized should not appear popular.
            if serde_json::from_str::<serde_json::Value>(&entry.response).is_err() {
                tracing::warn!("cache entry {} has corrupted response JSON; evicting", &key[..8]);
                let _ = self.delete_entry(&key).await;
                return Ok(None);
            }
            self.bump_hit(&key).await?;
            return Ok(Some(entry));
        }
        Ok(None)
    }

    /// Semantic lookup: returns entries in the same domain whose stored embedding
    /// is within `sim_threshold` cosine similarity of `query_embedding`.
    /// `embedding_model` filters out stale embeddings created by a different model;
    /// entries with an empty model string (pre-migration) are always included.
    pub async fn lookup_semantic(
        &self,
        domain: &str,
        query_embedding: &[f32],
        sim_threshold: f64,
        limit: usize,
        embedding_model: &str,
    ) -> Result<Vec<(CacheEntry, f64)>> {
        let now = Utc::now().timestamp();

        let rows = sqlx::query(
            "SELECT ce.id, ce.domain, ce.intent, ce.complexity, ce.prompt_text, ce.response,
                    ce.model_used, ce.confidence, ce.created_at, ce.expires_at, ce.hit_count,
                    ce.last_hit_at, ce.node_id, ce.shared, ce.pinned, emb.embedding, emb.model
             FROM cache_entries ce
             JOIN cache_embeddings emb ON emb.cache_id = ce.id
             WHERE ce.domain = ? AND (ce.expires_at IS NULL OR ce.expires_at > ?)
               AND (emb.model = '' OR emb.model = ?)
             ORDER BY ce.hit_count DESC
             LIMIT 500"
        )
        .bind(domain)
        .bind(now)
        .bind(embedding_model)
        .fetch_all(&self.pool)
        .await?;

        let mut results: Vec<(CacheEntry, f64)> = rows
            .into_iter()
            .filter_map(|row| {
                let blob: Vec<u8> = row.get("embedding");
                let stored = decode_embedding(&blob)?;
                let sim = cosine_similarity(query_embedding, &stored);
                if sim >= sim_threshold {
                    Some((row_to_entry_partial(&row), sim))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        for (entry, _) in &results {
            self.bump_hit(&entry.id).await?;
        }

        Ok(results)
    }

    /// Fetch candidate few-shot examples for local model augmentation.
    /// Returns entries in the same domain whose embedding falls in
    /// `[min_sim, max_sim)` — related enough to be informative, but below
    /// the cache-serve threshold so they wouldn't have been served directly.
    /// Does NOT bump hit counts; this is context injection, not cache serving.
    pub async fn lookup_fewshot(
        &self,
        domain:          &str,
        query_embedding: &[f32],
        min_sim:         f64,
        max_sim:         f64,
        limit:           usize,
        embedding_model: &str,
    ) -> Result<Vec<(CacheEntry, f64)>> {
        let now = Utc::now().timestamp();

        let rows = sqlx::query(
            "SELECT ce.id, ce.domain, ce.intent, ce.complexity, ce.prompt_text, ce.response,
                    ce.model_used, ce.confidence, ce.created_at, ce.expires_at, ce.hit_count,
                    ce.node_id, ce.shared, ce.pinned, emb.embedding
             FROM cache_entries ce
             JOIN cache_embeddings emb ON emb.cache_id = ce.id
             WHERE ce.domain = ?
               AND (ce.expires_at IS NULL OR ce.expires_at > ?)
               AND (emb.model = '' OR emb.model = ?)
             ORDER BY ce.hit_count DESC
             LIMIT 200"
        )
        .bind(domain)
        .bind(now)
        .bind(embedding_model)
        .fetch_all(&self.pool)
        .await?;

        let mut results: Vec<(CacheEntry, f64)> = rows
            .into_iter()
            .filter_map(|row| {
                let blob: Vec<u8> = row.get("embedding");
                let stored = decode_embedding(&blob)?;
                let sim = cosine_similarity(query_embedding, &stored);
                if sim >= min_sim && sim < max_sim {
                    Some((row_to_entry_partial(&row), sim))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    /// Returns the highest cosine similarity found for any live entry in this domain,
    /// regardless of threshold.  Does NOT bump hit counts — used only by the routing
    /// gate to distinguish "near miss" from "truly never seen."
    pub async fn best_semantic_sim(
        &self,
        domain: &str,
        query_embedding: &[f32],
        embedding_model: &str,
    ) -> Result<Option<f64>> {
        let now = Utc::now().timestamp();
        let rows = sqlx::query(
            "SELECT emb.embedding
             FROM cache_entries ce
             JOIN cache_embeddings emb ON emb.cache_id = ce.id
             WHERE ce.domain = ? AND (ce.expires_at IS NULL OR ce.expires_at > ?)
               AND (emb.model = '' OR emb.model = ?)
             LIMIT 500"
        )
        .bind(domain)
        .bind(now)
        .bind(embedding_model)
        .fetch_all(&self.pool)
        .await?;

        let best = rows.iter()
            .filter_map(|row| {
                let blob: Vec<u8> = row.get("embedding");
                decode_embedding(&blob).map(|stored| cosine_similarity(query_embedding, &stored))
            })
            .reduce(f64::max);

        Ok(best)
    }

    // ── Store ────────────────────────────────────────────────────────────

    pub async fn store(
        &self,
        shape:      &ShapeKey,
        prompt:     &str,
        system:     Option<&str>,
        response:   &str,
        model_used: &str,
        confidence: Option<f64>,
        ttl_secs:   Option<u64>,
        shared:     bool,
        pinned:     bool,
    ) -> Result<String> {
        let id         = Self::content_key(prompt, system);
        let now        = Utc::now().timestamp();
        let expires_at: Option<i64> = if pinned { None } else { ttl_secs.map(|t| now + t as i64) };

        sqlx::query(
            "INSERT INTO cache_entries
                (id, domain, intent, complexity, prompt_text, response, model_used,
                 confidence, created_at, expires_at, hit_count, node_id, shared, pinned)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
             ON CONFLICT(id) DO UPDATE SET
                response   = excluded.response,
                model_used = excluded.model_used,
                confidence = excluded.confidence,
                expires_at = excluded.expires_at,
                shared     = excluded.shared,
                pinned     = excluded.pinned"
        )
        .bind(&id)
        .bind(&shape.domain)
        .bind(&shape.intent)
        .bind(shape.complexity)
        .bind(prompt)
        .bind(response)
        .bind(model_used)
        .bind(confidence)
        .bind(now)
        .bind(expires_at)
        .bind(&self.node_id)
        .bind(shared as i32)
        .bind(pinned as i32)
        .execute(&self.pool)
        .await?;

        Ok(id)
    }

    /// Store an embedding vector alongside a cache entry.
    pub async fn store_embedding(
        &self,
        cache_id:  &str,
        embedding: &[f32],
        model:     &str,
    ) -> Result<()> {
        let blob = encode_embedding(embedding);
        sqlx::query(
            "INSERT INTO cache_embeddings (cache_id, embedding, model)
             VALUES (?, ?, ?)
             ON CONFLICT(cache_id) DO UPDATE SET embedding = excluded.embedding, model = excluded.model"
        )
        .bind(cache_id)
        .bind(blob)
        .bind(model)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── Entry management ─────────────────────────────────────────────────

    /// Hard-delete a cache entry by its content key.  Returns true if a row was removed.
    pub async fn delete_entry(&self, id: &str) -> Result<bool> {
        let r = sqlx::query("DELETE FROM cache_entries WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(r.rows_affected() > 0)
    }

    /// Pin or unpin an entry.  Pinned entries are never evicted.
    pub async fn set_pinned(&self, id: &str, pinned: bool) -> Result<bool> {
        let r = sqlx::query("UPDATE cache_entries SET pinned = ? WHERE id = ?")
            .bind(pinned as i32)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(r.rows_affected() > 0)
    }

    /// Browse / search cached entries.  Filters are all optional.
    /// Uses FTS5 full-text search when a query is provided; falls back to a
    /// full scan when no query is given (or if FTS5 is unavailable).
    pub async fn search_entries(
        &self,
        query:  Option<&str>,
        domain: Option<&str>,
        limit:  i64,
    ) -> Result<Vec<CacheEntrySummary>> {
        let now         = Utc::now().timestamp();
        let domain_flag = domain.unwrap_or("all");

        let rows = if let Some(q) = query.filter(|s| !s.trim().is_empty()) {
            // Sanitize query for FTS5: keep word chars and spaces, append prefix wildcard.
            let clean: String = q.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '_')
                .collect::<String>();
            let fts_term = format!("{}*", clean.trim());

            let result = sqlx::query(
                "SELECT ce.id, ce.domain, ce.intent, ce.prompt_text, ce.model_used,
                        ce.hit_count, ce.pinned, ce.created_at, ce.expires_at
                 FROM cache_entries_fts fts
                 JOIN cache_entries ce ON ce.id = fts.id
                 WHERE fts MATCH ?
                   AND (ce.expires_at IS NULL OR ce.expires_at > ?)
                   AND (? = 'all' OR ce.domain = ?)
                 ORDER BY ce.hit_count DESC
                 LIMIT ?"
            )
            .bind(&fts_term)
            .bind(now)
            .bind(domain_flag)
            .bind(domain.unwrap_or(""))
            .bind(limit)
            .fetch_all(&self.pool)
            .await;

            match result {
                Ok(rows) => rows,
                // FTS5 unavailable or query syntax error — fall back to LIKE.
                Err(_) => {
                    let like_pattern = format!("%{}%", q.to_lowercase());
                    sqlx::query(
                        "SELECT id, domain, intent, prompt_text, model_used, hit_count, pinned,
                                created_at, expires_at
                         FROM cache_entries
                         WHERE (expires_at IS NULL OR expires_at > ?)
                           AND (? = 'all' OR domain = ?)
                           AND LOWER(prompt_text) LIKE ?
                         ORDER BY hit_count DESC
                         LIMIT ?"
                    )
                    .bind(now)
                    .bind(domain_flag)
                    .bind(domain.unwrap_or(""))
                    .bind(&like_pattern)
                    .bind(limit)
                    .fetch_all(&self.pool)
                    .await?
                }
            }
        } else {
            sqlx::query(
                "SELECT id, domain, intent, prompt_text, model_used, hit_count, pinned,
                        created_at, expires_at
                 FROM cache_entries
                 WHERE (expires_at IS NULL OR expires_at > ?)
                   AND (? = 'all' OR domain = ?)
                 ORDER BY hit_count DESC
                 LIMIT ?"
            )
            .bind(now)
            .bind(domain_flag)
            .bind(domain.unwrap_or(""))
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };

        Ok(rows.into_iter().map(|row| {
            let text: String = row.get("prompt_text");
            CacheEntrySummary {
                id:             row.get("id"),
                domain:         row.get("domain"),
                intent:         row.get("intent"),
                prompt_preview: if text.len() > 200 { format!("{}…", truncate_chars(&text, 200)) } else { text },
                model_used:     row.get("model_used"),
                hit_count:      row.get("hit_count"),
                pinned:         row.get::<i32, _>("pinned") != 0,
                created_at:     row.get("created_at"),
                expires_at:     row.get("expires_at"),
            }
        }).collect())
    }

    // ── Domain knowledge (Layer 2) ───────────────────────────────────────

    /// Upsert a distilled knowledge document for a domain.  Version increments
    /// on every update so the portal can show how many distillation passes ran.
    pub async fn store_knowledge_doc(
        &self,
        domain:      &str,
        content:     &str,
        entry_count: usize,
    ) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO domain_knowledge (domain, content, entry_count, version, created_at, updated_at)
             VALUES (?, ?, ?, 1, ?, ?)
             ON CONFLICT(domain) DO UPDATE SET
                 content     = excluded.content,
                 entry_count = excluded.entry_count,
                 version     = version + 1,
                 updated_at  = excluded.updated_at"
        )
        .bind(domain)
        .bind(content)
        .bind(entry_count as i64)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Load the current knowledge document for a domain, if one exists.
    pub async fn load_knowledge_doc(&self, domain: &str) -> Result<Option<String>> {
        let row = sqlx::query("SELECT content FROM domain_knowledge WHERE domain = ?")
            .bind(domain)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.get("content")))
    }

    /// List all stored knowledge documents — used by the management portal.
    pub async fn list_knowledge_docs(&self) -> Result<Vec<KnowledgeDocSummary>> {
        let rows = sqlx::query(
            "SELECT domain, entry_count, version, updated_at
             FROM domain_knowledge ORDER BY updated_at DESC"
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|r| KnowledgeDocSummary {
            domain:      r.get("domain"),
            entry_count: r.get("entry_count"),
            version:     r.get("version"),
            updated_at:  r.get("updated_at"),
        }).collect())
    }

    /// Returns domains that have enough live cache entries to be worth distilling.
    pub async fn distillation_candidates(&self, min_entries: i64) -> Result<Vec<String>> {
        let now = Utc::now().timestamp();
        let rows = sqlx::query(
            "SELECT domain FROM cache_entries
             WHERE (expires_at IS NULL OR expires_at > ?)
             GROUP BY domain
             HAVING COUNT(*) >= ?
             ORDER BY COUNT(*) DESC"
        )
        .bind(now)
        .bind(min_entries)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(|r| r.get("domain")).collect())
    }

    /// Return the top `limit` prompt+answer pairs for a domain, ranked by hit count.
    /// Answers are extracted from stored response JSON on the fly.
    pub async fn top_entries_for_domain(
        &self,
        domain: &str,
        limit:  i64,
    ) -> Result<Vec<(String, String)>> {
        let now = Utc::now().timestamp();
        let rows = sqlx::query(
            "SELECT prompt_text, response FROM cache_entries
             WHERE domain = ? AND (expires_at IS NULL OR expires_at > ?)
             ORDER BY hit_count DESC
             LIMIT ?"
        )
        .bind(domain)
        .bind(now)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().filter_map(|r| {
            let prompt:   String = r.get("prompt_text");
            let response: String = r.get("response");
            let answer = extract_answer_from_response_json(&response)?;
            Some((prompt, answer))
        }).collect())
    }

    // ── Adaptive thresholds (Layer 3) ────────────────────────────────────

    /// Upsert an adaptive novelty threshold override for a (domain, intent) pair.
    pub async fn store_threshold_override(
        &self,
        domain:          &str,
        intent:          &str,
        novelty_override: f64,
        escalation_rate: f64,
        sample_count:    i64,
    ) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO routing_thresholds
                 (domain, intent, novelty_override, escalation_rate, sample_count, computed_at)
             VALUES (?, ?, ?, ?, ?, ?)
             ON CONFLICT(domain, intent) DO UPDATE SET
                 novelty_override = excluded.novelty_override,
                 escalation_rate  = excluded.escalation_rate,
                 sample_count     = excluded.sample_count,
                 computed_at      = excluded.computed_at"
        )
        .bind(domain)
        .bind(intent)
        .bind(novelty_override)
        .bind(escalation_rate)
        .bind(sample_count)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Load all stored threshold overrides as a `(domain, intent) → novelty_threshold` map.
    /// Called once at startup to seed the in-memory ArcSwap.
    pub async fn load_threshold_overrides(
        &self,
    ) -> Result<std::collections::HashMap<(String, String), f64>> {
        let rows = sqlx::query(
            "SELECT domain, intent, novelty_override FROM routing_thresholds"
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|r| {
            let domain: String = r.get("domain");
            let intent: String = r.get("intent");
            let val:    f64    = r.get("novelty_override");
            ((domain, intent), val)
        }).collect())
    }

    /// Compute per-(domain, intent) escalation rates over the last `window_secs`.
    /// Only considers requests that actually reached local model routing
    /// (excludes cache hits, routing gate rejections, tool-use, user-direct bypasses).
    pub async fn escalation_stats(
        &self,
        window_secs: i64,
        min_samples: i64,
    ) -> Result<Vec<EscalationStat>> {
        let now   = Utc::now().timestamp();
        let since = now - window_secs;

        let rows = sqlx::query(
            "SELECT domain, intent,
                    COUNT(*) AS total,
                    SUM(CASE WHEN decision = 'api' THEN 1 ELSE 0 END) AS escalations
             FROM routing_log
             WHERE created_at > ? AND created_at <= ?
               AND domain IS NOT NULL
               AND intent  IS NOT NULL
               AND decision IN ('local', 'api')
             GROUP BY domain, intent
             HAVING total >= ?
             ORDER BY escalations DESC"
        )
        .bind(since)
        .bind(now)
        .bind(min_samples)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| {
            let total:      i64 = r.get("total");
            let escalations: i64 = r.get("escalations");
            EscalationStat {
                domain:          r.get("domain"),
                intent:          r.get("intent"),
                escalation_rate: if total > 0 { escalations as f64 / total as f64 } else { 0.0 },
                sample_count:    total,
            }
        }).collect())
    }

    // ── Layer 4: Response quality feedback ───────────────────────────────

    /// Record an explicit quality signal (`"good"` or `"bad"`) from a user
    /// annotation, or an implicit `"repeat"` from repeat-detection.
    pub async fn record_feedback(
        &self,
        domain: &str,
        intent: &str,
        signal: &str,
        source: &str,
    ) -> Result<()> {
        let id  = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO response_feedback (id, domain, intent, signal, source, created_at)
             VALUES (?, ?, ?, ?, ?, ?)"
        )
        .bind(&id)
        .bind(domain)
        .bind(intent)
        .bind(signal)
        .bind(source)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Aggregate per-(domain, intent) quality signal counts within `window_secs`.
    /// Called by `ThresholdAdaptor::adapt()` to blend satisfaction signals into
    /// the threshold calibration alongside implicit escalation rates.
    pub async fn quality_stats(&self, window_secs: i64) -> Result<Vec<QualityStat>> {
        let since = Utc::now().timestamp() - window_secs;
        let rows = sqlx::query(
            "SELECT domain, intent,
                    SUM(CASE WHEN signal = 'bad'  THEN 1 ELSE 0 END) AS bad_count,
                    SUM(CASE WHEN signal = 'good' THEN 1 ELSE 0 END) AS good_count
             FROM response_feedback
             WHERE created_at > ?
             GROUP BY domain, intent"
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| QualityStat {
            domain:    r.get("domain"),
            intent:    r.get("intent"),
            bad_count:  r.get("bad_count"),
            good_count: r.get("good_count"),
        }).collect())
    }

    /// Return the most recent feedback rows for the learning portal.
    pub async fn list_recent_feedback(&self, limit: i64) -> Result<Vec<serde_json::Value>> {
        let rows = sqlx::query(
            "SELECT domain, intent, signal, source, created_at
             FROM response_feedback
             ORDER BY created_at DESC LIMIT ?"
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| serde_json::json!({
            "domain":     r.get::<String, _>("domain"),
            "intent":     r.get::<String, _>("intent"),
            "signal":     r.get::<String, _>("signal"),
            "source":     r.get::<String, _>("source"),
            "created_at": r.get::<i64, _>("created_at"),
        })).collect())
    }

    // ── Layer 5: Contrastive failure pairs ───────────────────────────────

    /// Store a (wrong attempt, correct answer) pair captured at escalation time.
    /// `local_attempt` is the text the local model generated before confidence
    /// fell below the floor; `correct_answer` is the text from the API response.
    pub async fn store_escalation_pair(
        &self,
        cache_id:         Option<&str>,
        domain:           &str,
        intent:           &str,
        prompt_text:      &str,
        local_attempt:    &str,
        correct_answer:   &str,
        local_confidence: Option<f64>,
    ) -> Result<String> {
        let id  = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO escalation_pairs
             (id, cache_id, domain, intent, prompt_text, local_attempt, correct_answer, local_confidence, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&id)
        .bind(cache_id)
        .bind(domain)
        .bind(intent)
        .bind(prompt_text)
        .bind(local_attempt)
        .bind(correct_answer)
        .bind(local_confidence)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    /// Return up to `limit` recent contrast pairs for a domain, newest first.
    /// Used by the distiller to include in the synthesis prompt, and by
    /// few-shot injection when `contrast_in_fewshot` is enabled.
    pub async fn contrast_pairs_for_domain(
        &self,
        domain: &str,
        limit:  i64,
    ) -> Result<Vec<ContrastPair>> {
        let rows = sqlx::query(
            "SELECT id, domain, intent, prompt_text, local_attempt, correct_answer, local_confidence, created_at
             FROM escalation_pairs
             WHERE domain = ?
             ORDER BY created_at DESC LIMIT ?"
        )
        .bind(domain)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| ContrastPair {
            id:               r.get("id"),
            domain:           r.get("domain"),
            intent:           r.get("intent"),
            prompt_text:      r.get("prompt_text"),
            local_attempt:    r.get("local_attempt"),
            correct_answer:   r.get("correct_answer"),
            local_confidence: r.get("local_confidence"),
            created_at:       r.get("created_at"),
        }).collect())
    }

    /// Return the most recent contrast pairs across all domains for the portal.
    pub async fn list_recent_contrasts(&self, limit: i64) -> Result<Vec<ContrastPair>> {
        let rows = sqlx::query(
            "SELECT id, domain, intent, prompt_text, local_attempt, correct_answer, local_confidence, created_at
             FROM escalation_pairs
             ORDER BY created_at DESC LIMIT ?"
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| ContrastPair {
            id:               r.get("id"),
            domain:           r.get("domain"),
            intent:           r.get("intent"),
            prompt_text:      r.get("prompt_text"),
            local_attempt:    r.get("local_attempt"),
            correct_answer:   r.get("correct_answer"),
            local_confidence: r.get("local_confidence"),
            created_at:       r.get("created_at"),
        }).collect())
    }

    // ── Confidence calibration (Layer 6) ─────────────────────────────────

    /// Store one calibration observation: the local model claimed `claimed_conf`
    /// and the actual word-overlap similarity with the API answer was `actual_sim`.
    pub async fn store_calibration_sample(
        &self,
        domain:      &str,
        intent:      &str,
        claimed_conf: f64,
        actual_sim:  f64,
    ) -> Result<()> {
        let id  = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO calibration_log (id, domain, intent, claimed_conf, actual_sim, created_at)
             VALUES (?, ?, ?, ?, ?, ?)"
        )
        .bind(&id)
        .bind(domain)
        .bind(intent)
        .bind(claimed_conf)
        .bind(actual_sim)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Compute per-(domain, intent) mean calibration bias: `mean(actual_sim − claimed_conf)`.
    /// A negative bias means the model overclaims confidence; positive means underconfidence.
    pub async fn load_calibration_biases(
        &self,
        window_secs: i64,
    ) -> Result<std::collections::HashMap<(String, String), f64>> {
        let since = Utc::now().timestamp() - window_secs;
        let rows = sqlx::query(
            "SELECT domain, intent, AVG(actual_sim - claimed_conf) AS bias, COUNT(*) AS n
             FROM calibration_log
             WHERE created_at > ?
             GROUP BY domain, intent
             HAVING n >= 3"
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().filter_map(|r| {
            let bias: f64 = r.get("bias");
            if bias.is_nan() || bias.is_infinite() { return None; }
            Some(((r.get::<String, _>("domain"), r.get::<String, _>("intent")), bias))
        }).collect())
    }

    /// Sample up to `limit` non-expired cache entries that were generated by the
    /// Anthropic API (model_used = 'anthropic'), chosen at random.  Used by the
    /// CalibrationRunner to build its test set.
    pub async fn sample_api_entries(&self, limit: usize) -> Result<Vec<CacheEntry>> {
        let now = Utc::now().timestamp();
        let rows = sqlx::query(
            "SELECT id, domain, intent, complexity, prompt_text, response, model_used,
                    confidence, created_at, expires_at, hit_count, node_id, shared, pinned
             FROM cache_entries
             WHERE model_used = 'anthropic'
               AND (expires_at IS NULL OR expires_at > ?)
             ORDER BY RANDOM() LIMIT ?"
        )
        .bind(now)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(row_to_entry).collect())
    }

    // ── Federation helpers ───────────────────────────────────────────────

    /// Look up a cache entry by its raw SHA256 hash — used by federation peers.
    pub async fn lookup_by_hash(&self, hash: &str) -> Result<Option<CacheEntry>> {
        let now = Utc::now().timestamp();
        let row = sqlx::query(
            "SELECT id, domain, intent, complexity, prompt_text, response, model_used,
                    confidence, created_at, expires_at, hit_count, node_id, shared, pinned
             FROM cache_entries
             WHERE id = ? AND shared = 1 AND (expires_at IS NULL OR expires_at > ?)"
        )
        .bind(hash)
        .bind(now)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(row_to_entry))
    }

    /// Return a page of shared, non-expired hashes — used to gossip our inventory to peers.
    pub async fn list_shared_hashes(&self, limit: i64, offset: i64) -> Result<Vec<String>> {
        let now = Utc::now().timestamp();
        let rows = sqlx::query(
            "SELECT id FROM cache_entries \
             WHERE shared = 1 AND (expires_at IS NULL OR expires_at > ?) \
             ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        .bind(now)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(|r| r.get::<String, _>("id")).collect())
    }

    // ── Stats ────────────────────────────────────────────────────────────

    pub async fn stats(&self) -> Result<CacheStats> {
        let total: i64 = sqlx::query("SELECT COUNT(*) as c FROM cache_entries")
            .fetch_one(&self.pool).await?.get("c");
        let hits: i64 = sqlx::query("SELECT COALESCE(SUM(hit_count), 0) as h FROM cache_entries")
            .fetch_one(&self.pool).await?.get("h");
        let shared: i64 = sqlx::query("SELECT COUNT(*) as c FROM cache_entries WHERE shared = 1")
            .fetch_one(&self.pool).await?.get("c");

        let db_bytes  = std::fs::metadata(&self.db_path).map(|m| m.len()).unwrap_or(0) as i64;
        let wal_bytes = std::fs::metadata(format!("{}-wal", self.db_path)).map(|m| m.len()).unwrap_or(0) as i64;

        Ok(CacheStats {
            total_entries:  total,
            total_hits:     hits,
            shared_entries: shared,
            db_size_bytes:  db_bytes + wal_bytes,
        })
    }

    pub async fn evict_to_size_limit(&self, max_bytes: u64) -> Result<u64> {
        let mut total_evicted = 0u64;
        let mut cap_reached   = true;
        for _ in 0..200 {
            let row = sqlx::query(
                "SELECT (page_count - freelist_count) * page_size AS live \
                 FROM pragma_page_count(), pragma_freelist_count(), pragma_page_size()"
            ).fetch_one(&self.pool).await?;
            let live_bytes: i64 = row.get("live");
            if (live_bytes as u64) <= max_bytes { cap_reached = false; break; }

            let r = sqlx::query(
                "DELETE FROM cache_entries WHERE id IN (
                     SELECT id FROM cache_entries
                     WHERE pinned = 0
                     ORDER BY COALESCE(last_hit_at, created_at) ASC
                     LIMIT 100
                 )"
            ).execute(&self.pool).await?;
            let deleted = r.rows_affected();
            if deleted == 0 {
                // No more unpinned entries to evict — everything left is pinned.
                if (live_bytes as u64) > max_bytes {
                    tracing::warn!(
                        "cache is {} MB (limit {} MB) but all remaining entries are pinned; \
                         cannot evict further. Unpin entries or raise [cache] max_size_mb.",
                        live_bytes / 1_048_576,
                        max_bytes  / 1_048_576,
                    );
                }
                cap_reached = false;
                break;
            }
            total_evicted += deleted;
        }

        if cap_reached {
            if let Ok(row) = sqlx::query(
                "SELECT (page_count - freelist_count) * page_size AS live \
                 FROM pragma_page_count(), pragma_freelist_count(), pragma_page_size()"
            ).fetch_one(&self.pool).await {
                let live_bytes: i64 = row.get("live");
                if (live_bytes as u64) > max_bytes {
                    tracing::warn!(
                        "cache size-limit eviction hit the 200-iteration cap: \
                         cache is {} MB but limit is {} MB — \
                         consider raising [cache] max_size_mb or pinning fewer entries",
                        live_bytes / 1_048_576,
                        max_bytes  / 1_048_576
                    );
                }
            }
        }

        Ok(total_evicted)
    }

    pub async fn log_routing(
        &self,
        shape_key:   &str,
        domain:      &str,
        intent:      &str,
        decision:    &str,
        backend:     &str,
        latency_ms:  i64,
        tokens_in:   Option<i64>,
        tokens_out:  Option<i64>,
        saved_usd:   Option<f64>,
        miss_reason: Option<&str>,
        scores_json: Option<&str>,
    ) -> Result<()> {
        let id  = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO routing_log (id, shape_key, domain, intent, decision, backend,
                                      latency_ms, tokens_in, tokens_out, saved_usd,
                                      miss_reason, scores_json, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&id)
        .bind(shape_key)
        .bind(domain)
        .bind(intent)
        .bind(decision)
        .bind(backend)
        .bind(latency_ms)
        .bind(tokens_in)
        .bind(tokens_out)
        .bind(saved_usd)
        .bind(miss_reason)
        .bind(scores_json)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn purge_by_node_id(&self, node_id: &str) -> Result<u64> {
        let r = sqlx::query("DELETE FROM cache_entries WHERE node_id = ?")
            .bind(node_id)
            .execute(&self.pool)
            .await?;
        Ok(r.rows_affected())
    }

    pub async fn domain_hit_count(&self, domain: &str, intent: &str) -> Result<i64> {
        let now       = Utc::now().timestamp();
        let thirty_days_ago = now - 2_592_000; // 30 days

        // Part A: active cache entries still within TTL
        let cache_hits: i64 = sqlx::query(
            "SELECT COALESCE(SUM(hit_count), 0) AS total \
             FROM cache_entries \
             WHERE domain = ? AND intent = ? \
               AND (expires_at IS NULL OR expires_at > ?)"
        )
        .bind(domain)
        .bind(intent)
        .bind(now)
        .fetch_one(&self.pool)
        .await?
        .get("total");

        // Part B: routing_log history — successful local/cache decisions survive expiry
        let log_hits: i64 = sqlx::query(
            "SELECT COUNT(*) AS total \
             FROM routing_log \
             WHERE domain = ? AND intent = ? \
               AND decision IN ('exact_cache','semantic_cache','local','federation') \
               AND created_at > ?"
        )
        .bind(domain)
        .bind(intent)
        .bind(thirty_days_ago)
        .fetch_one(&self.pool)
        .await?
        .get("total");

        Ok(cache_hits + log_hits)
    }

    pub async fn routing_log_recent(&self, limit: i64) -> Result<Vec<RoutingLogEntry>> {
        let rows = sqlx::query(
            "SELECT shape_key, domain, intent, decision, backend, latency_ms, tokens_in,
                    tokens_out, saved_usd, miss_reason, created_at
             FROM routing_log ORDER BY created_at DESC LIMIT ?"
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| RoutingLogEntry {
            shape_key:   r.get("shape_key"),
            domain:      r.get("domain"),
            intent:      r.get("intent"),
            decision:    r.get("decision"),
            backend:     r.get("backend"),
            latency_ms:  r.get("latency_ms"),
            tokens_in:   r.get("tokens_in"),
            tokens_out:  r.get("tokens_out"),
            saved_usd:   r.get("saved_usd"),
            miss_reason: r.get("miss_reason"),
            created_at:  r.get("created_at"),
        }).collect())
    }

    pub async fn routing_log_stats(&self, since_secs: i64) -> Result<serde_json::Value> {
        use serde_json::json;
        let since = Utc::now().timestamp() - since_secs;

        let rows = sqlx::query(
            "SELECT decision, COUNT(*) as cnt,
                    AVG(CAST(latency_ms AS REAL)) as avg_lat,
                    COALESCE(SUM(saved_usd), 0.0) as saved
             FROM routing_log WHERE created_at >= ?
             GROUP BY decision ORDER BY cnt DESC"
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        let total: i64 = rows.iter().map(|r| r.get::<i64, _>("cnt")).sum();

        let by_decision: Vec<serde_json::Value> = rows.iter().map(|r| {
            let cnt: i64     = r.get("cnt");
            let avg_lat: f64 = r.get("avg_lat");
            let saved: f64   = r.get("saved");
            json!({
                "decision":       r.get::<String, _>("decision"),
                "count":          cnt,
                "pct":            if total > 0 { cnt as f64 / total as f64 * 100.0 } else { 0.0 },
                "avg_latency_ms": avg_lat,
                "saved_usd":      saved,
            })
        }).collect();

        // Miss reason breakdown — only for rows that went to API/local (not cache hits).
        let miss_rows = sqlx::query(
            "SELECT miss_reason, COUNT(*) as cnt
             FROM routing_log
             WHERE created_at >= ? AND miss_reason IS NOT NULL
             GROUP BY miss_reason ORDER BY cnt DESC"
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        let miss_total: i64 = miss_rows.iter().map(|r| r.get::<i64, _>("cnt")).sum();
        let by_miss_reason: Vec<serde_json::Value> = miss_rows.iter().map(|r| {
            let cnt: i64 = r.get("cnt");
            json!({
                "reason": r.get::<Option<String>, _>("miss_reason"),
                "count":  cnt,
                "pct":    if miss_total > 0 { cnt as f64 / miss_total as f64 * 100.0 } else { 0.0 },
            })
        }).collect();

        Ok(json!({
            "total_requests":  total,
            "by_decision":     by_decision,
            "by_miss_reason":  by_miss_reason,
            "window_hours":    since_secs / 3600,
        }))
    }

    /// Export cache entries for backup or cross-node seeding.
    /// Returns full entries including the raw response payload.
    /// Filters: domain (exact), pinned-only, active-only (default: excludes expired).
    pub async fn export_entries(
        &self,
        domain:      Option<&str>,
        pinned_only: bool,
        limit:       i64,
    ) -> Result<Vec<CacheExportEntry>> {
        let now          = Utc::now().timestamp();
        let domain_flag  = domain.unwrap_or("all");
        let pinned_flag  = if pinned_only { 1i32 } else { 0i32 };

        let rows = sqlx::query(
            "SELECT id, domain, intent, prompt_text, response, model_used,
                    hit_count, pinned, created_at, expires_at
             FROM cache_entries
             WHERE (expires_at IS NULL OR expires_at > ?)
               AND (? = 'all' OR domain = ?)
               AND (? = 0 OR pinned = 1)
             ORDER BY hit_count DESC, created_at DESC
             LIMIT ?"
        )
        .bind(now)
        .bind(domain_flag)
        .bind(domain.unwrap_or(""))
        .bind(pinned_flag)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|row| CacheExportEntry {
            id:          row.get("id"),
            domain:      row.get("domain"),
            intent:      row.get("intent"),
            prompt_text: row.get("prompt_text"),
            response:    row.get("response"),
            model_used:  row.get("model_used"),
            hit_count:   row.get("hit_count"),
            pinned:      row.get::<i32, _>("pinned") != 0,
            created_at:  row.get("created_at"),
            expires_at:  row.get("expires_at"),
        }).collect())
    }

    /// Evict expired entries (respects pinned — pinned entries never expire).
    pub async fn evict_expired(&self) -> Result<u64> {
        let now = Utc::now().timestamp();
        let r = sqlx::query(
            "DELETE FROM cache_entries
             WHERE pinned = 0 AND expires_at IS NOT NULL AND expires_at <= ?"
        )
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(r.rows_affected())
    }

    async fn bump_hit(&self, id: &str) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            "UPDATE cache_entries \
             SET hit_count = MIN(hit_count + 1, 9223372036854775807), last_hit_at = ? \
             WHERE id = ?"
        )
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── Forgetting curves ─────────────────────────────────────────────────────

    /// Apply Ebbinghaus-style forgetting curves to cache TTLs.
    ///
    /// For each non-pinned entry that has an explicit expiry, computes:
    ///   `strength  = 1 + ln(1 + hit_count)` — grows logarithmically with use
    ///   `new_ttl   = base_ttl * min(strength, max_multiplier)`
    ///   `new_expiry = COALESCE(last_hit_at, created_at) + new_ttl`
    ///
    /// High-use entries that keep being accessed have their `last_hit_at`
    /// refreshed on every hit, so the new expiry keeps advancing.  Entries
    /// that go unaccessed have a stale `last_hit_at`, so the new expiry
    /// naturally falls sooner than a fixed-TTL scheme would allow.
    ///
    /// Only updates rows where the new expiry differs by more than one hour
    /// to avoid noisy writes.  Returns the count of rows updated.
    pub async fn adjust_ttl_forgetting(
        &self,
        default_ttl_secs: u64,
        domain_ttls: &std::collections::HashMap<String, u64>,
        max_multiplier: f64,
    ) -> Result<usize> {
        let now = Utc::now().timestamp();

        let rows = sqlx::query(
            "SELECT id, domain, hit_count, created_at, last_hit_at, expires_at
             FROM cache_entries
             WHERE pinned = 0 AND expires_at IS NOT NULL AND expires_at > ?
             LIMIT 5000"
        )
        .bind(now)
        .fetch_all(&self.pool)
        .await?;

        let mut updated = 0usize;
        let mut tx = self.pool.begin().await?;
        for row in &rows {
            let id:          String      = row.get("id");
            let domain:      String      = row.get("domain");
            let hit_count:   i64         = row.get("hit_count");
            let created_at:  i64         = row.get("created_at");
            let last_hit_at: Option<i64> = row.get("last_hit_at");
            let current_exp: i64         = row.get("expires_at");

            let base_ttl = domain_ttls.get(&domain).copied()
                .unwrap_or(default_ttl_secs) as f64;

            let strength   = (1.0_f64 + (1.0_f64 + hit_count as f64).ln())
                .min(max_multiplier);
            let new_ttl    = (base_ttl * strength) as i64;
            let anchor     = last_hit_at.unwrap_or(created_at);
            let new_expiry = anchor + new_ttl;

            if (new_expiry - current_exp).abs() > 3600 {
                sqlx::query("UPDATE cache_entries SET expires_at = ? WHERE id = ?")
                    .bind(new_expiry)
                    .bind(&id)
                    .execute(&mut *tx)
                    .await?;
                updated += 1;
            }
        }
        tx.commit().await?;
        Ok(updated)
    }

    // ── Learning observability ────────────────────────────────────────────────

    /// Per-(domain, intent) calibration bias summary for the portal.
    /// Returns bias, sample count, and whether the bias is significant (≥ 3 samples).
    pub async fn calibration_summary(&self, window_secs: i64) -> Result<serde_json::Value> {
        use serde_json::json;
        let since = Utc::now().timestamp() - window_secs;
        let rows = sqlx::query(
            "SELECT domain, intent,
                    AVG(actual_sim - claimed_conf) AS bias,
                    AVG(claimed_conf)              AS avg_claimed,
                    AVG(actual_sim)                AS avg_actual,
                    COUNT(*)                       AS n
             FROM calibration_log
             WHERE created_at > ?
             GROUP BY domain, intent
             ORDER BY n DESC, ABS(AVG(actual_sim - claimed_conf)) DESC"
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        let total_samples: i64 = rows.iter().map(|r| r.get::<i64, _>("n")).sum();
        let entries: Vec<serde_json::Value> = rows.iter().map(|r| {
            let bias: f64 = r.get("bias");
            json!({
                "domain":       r.get::<String, _>("domain"),
                "intent":       r.get::<String, _>("intent"),
                "bias":         bias,
                "avg_claimed":  r.get::<f64, _>("avg_claimed"),
                "avg_actual":   r.get::<f64, _>("avg_actual"),
                "samples":      r.get::<i64, _>("n"),
                "significant":  r.get::<i64, _>("n") >= 3 && !bias.is_nan(),
            })
        }).collect();

        Ok(json!({
            "window_hours":   window_secs / 3600,
            "total_samples":  total_samples,
            "by_domain_intent": entries,
        }))
    }

    /// Draft-verify hit rate and token savings from the routing log.
    pub async fn draft_verify_stats(&self, window_secs: i64) -> Result<serde_json::Value> {
        use serde_json::json;
        let since = Utc::now().timestamp() - window_secs;

        let total_api: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM routing_log WHERE created_at >= ? AND decision = 'api'"
        )
        .bind(since)
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let dv_row = sqlx::query(
            "SELECT COUNT(*) AS cnt,
                    COALESCE(AVG(latency_ms), 0.0)   AS avg_lat,
                    COALESCE(AVG(tokens_out), 0.0)   AS avg_out,
                    COALESCE(AVG(tokens_in),  0.0)   AS avg_in
             FROM routing_log
             WHERE created_at >= ? AND miss_reason = 'draft_verify'"
        )
        .bind(since)
        .fetch_one(&self.pool)
        .await?;

        let dv_count: i64 = dv_row.get("cnt");
        let avg_lat: f64  = dv_row.get("avg_lat");
        let avg_out: f64  = dv_row.get("avg_out");
        let avg_in:  f64  = dv_row.get("avg_in");

        let hit_rate = if total_api > 0 {
            dv_count as f64 / total_api as f64 * 100.0
        } else { 0.0 };

        Ok(json!({
            "window_hours":    window_secs / 3600,
            "total_api_calls": total_api,
            "draft_verify_hits": dv_count,
            "hit_rate_pct":    hit_rate,
            "avg_latency_ms":  avg_lat,
            "avg_tokens_in":   avg_in,
            "avg_tokens_out":  avg_out,
        }))
    }

    /// Distribution of live entries across forgetting-curve strength tiers.
    /// `max_multiplier` should match `cfg.cache.forgetting_max_multiplier` so the
    /// reported strength caps align with what `adjust_ttl_forgetting` actually applies.
    pub async fn forgetting_stats(&self, max_multiplier: f64) -> Result<serde_json::Value> {
        use serde_json::json;
        let now = Utc::now().timestamp();

        let rows = sqlx::query(
            "SELECT hit_count,
                    COUNT(*)            AS cnt,
                    AVG(
                        CASE WHEN expires_at IS NOT NULL
                             THEN CAST(expires_at - ? AS REAL)
                             ELSE NULL END
                    ) AS avg_remaining_secs
             FROM cache_entries
             WHERE (expires_at IS NULL OR expires_at > ?)
               AND pinned = 0
             GROUP BY hit_count
             ORDER BY hit_count"
        )
        .bind(now)   // SELECT: expires_at - ? = actual seconds until expiry
        .bind(now)   // WHERE:  expires_at > ?
        .fetch_all(&self.pool)
        .await?;

        let tiers: Vec<serde_json::Value> = rows.iter().map(|r| {
            let hits: i64 = r.get("hit_count");
            let strength  = (1.0_f64 + (1.0_f64 + hits as f64).ln()).min(max_multiplier);
            json!({
                "hit_count":           hits,
                "strength":            (strength * 100.0).round() / 100.0,
                "entry_count":         r.get::<i64, _>("cnt"),
                "avg_remaining_secs":  r.get::<Option<f64>, _>("avg_remaining_secs"),
            })
        }).collect();

        let total_live: i64 = rows.iter().map(|r| r.get::<i64, _>("cnt")).sum();

        Ok(json!({
            "total_live_entries": total_live,
            "max_multiplier":     max_multiplier,
            "tiers": tiers,
        }))
    }

    // ── Graph data ────────────────────────────────────────────────────────────

    /// D3-format nodes + links for the knowledge graph visualiser.
    pub async fn graph_data(&self, window_secs: i64) -> Result<serde_json::Value> {
        use serde_json::json;
        use std::collections::HashMap;
        let since = Utc::now().timestamp() - window_secs;

        let now_ts = Utc::now().timestamp();
        let entry_rows = sqlx::query(
            "SELECT domain, intent, COUNT(*) AS cnt
             FROM cache_entries
             WHERE (expires_at IS NULL OR expires_at > ?)
             GROUP BY domain, intent"
        )
        .bind(now_ts)
        .fetch_all(&self.pool).await.unwrap_or_default();

        let route_rows = sqlx::query(
            "SELECT domain, intent, COUNT(*) AS total,
                    SUM(CASE WHEN decision='api' THEN 1 ELSE 0 END) AS api_calls
             FROM routing_log WHERE created_at >= ? AND domain IS NOT NULL
             GROUP BY domain, intent"
        ).bind(since).fetch_all(&self.pool).await.unwrap_or_default();

        let doc_rows = sqlx::query(
            "SELECT domain, version, length(content) AS chars, entry_count
             FROM domain_knowledge"
        ).fetch_all(&self.pool).await.unwrap_or_default();

        let thresh_rows = sqlx::query(
            "SELECT domain, intent, novelty_override, escalation_rate
             FROM routing_thresholds"
        ).fetch_all(&self.pool).await.unwrap_or_default();

        let contrast_rows = sqlx::query(
            "SELECT domain, COUNT(*) AS count FROM escalation_pairs GROUP BY domain"
        ).fetch_all(&self.pool).await.unwrap_or_default();

        let feedback_rows = sqlx::query(
            "SELECT domain, signal, COUNT(*) AS count FROM response_feedback
             WHERE created_at >= ? GROUP BY domain, signal"
        ).bind(since).fetch_all(&self.pool).await.unwrap_or_default();

        // Accumulate domain-level entry totals
        let mut domain_totals: HashMap<String, i64> = HashMap::new();
        for r in &entry_rows {
            *domain_totals.entry(r.get::<String, _>("domain")).or_insert(0) += r.get::<i64, _>("cnt");
        }

        let mut nodes: Vec<serde_json::Value> = Vec::new();
        let mut links: Vec<serde_json::Value> = Vec::new();

        // Intent nodes
        for r in &entry_rows {
            let domain: String = r.get("domain");
            let intent: String = r.get("intent");
            let cnt: i64       = r.get("cnt");
            let (total_24h, api_24h) = route_rows.iter()
                .find(|rr| rr.get::<String,_>("domain") == domain && rr.get::<String,_>("intent") == intent)
                .map(|rr| (rr.get::<i64,_>("total"), rr.get::<i64,_>("api_calls")))
                .unwrap_or((0, 0));
            let esc = if total_24h > 0 { api_24h as f64 / total_24h as f64 } else { 0.5 };
            let (thr, base) = thresh_rows.iter()
                .find(|tr| tr.get::<String,_>("domain") == domain && tr.get::<String,_>("intent") == intent)
                .map(|tr| (tr.get::<f64,_>("novelty_override"), 0.60_f64))
                .unwrap_or((0.60, 0.60));
            let iid = format!("intent:{}:{}", domain, intent);
            nodes.push(json!({
                "id": iid, "type": "intent",
                "label": intent, "domain": domain.clone(),
                "entries": cnt, "escalation_rate": esc,
                "threshold": thr, "base_threshold": base,
                "adapted": (thr - base).abs() > 1e-9,
                "requests_24h": total_24h,
            }));
            links.push(json!({ "source": format!("domain:{}", domain), "target": iid, "value": cnt }));
        }

        // Domain nodes
        for (domain, entries) in &domain_totals {
            let doc = doc_rows.iter().find(|d| d.get::<String,_>("domain") == *domain);
            let contrast: i64 = contrast_rows.iter()
                .find(|c| c.get::<String,_>("domain") == *domain)
                .map(|c| c.get("count")).unwrap_or(0);
            let good: i64 = feedback_rows.iter()
                .filter(|f| f.get::<String,_>("domain") == *domain && f.get::<String,_>("signal") == "good")
                .map(|f| f.get::<i64,_>("count")).sum();
            let bad: i64 = feedback_rows.iter()
                .filter(|f| f.get::<String,_>("domain") == *domain && f.get::<String,_>("signal") == "bad")
                .map(|f| f.get::<i64,_>("count")).sum();
            let total_24h: i64 = route_rows.iter()
                .filter(|rr| rr.get::<Option<String>,_>("domain").as_deref() == Some(domain.as_str()))
                .map(|rr| rr.get::<i64,_>("total")).sum();
            let api_24h: i64 = route_rows.iter()
                .filter(|rr| rr.get::<Option<String>,_>("domain").as_deref() == Some(domain.as_str()))
                .map(|rr| rr.get::<i64,_>("api_calls")).sum();
            let esc = if total_24h > 0 { api_24h as f64 / total_24h as f64 } else { 0.5 };
            nodes.push(json!({
                "id": format!("domain:{}", domain), "type": "domain",
                "label": domain, "entries": entries,
                "escalation_rate": esc,
                "has_doc": doc.is_some(),
                "doc_chars": doc.map(|d| d.get::<i64,_>("chars")).unwrap_or(0),
                "doc_version": doc.map(|d| d.get::<i64,_>("version")).unwrap_or(0),
                "contrast_pairs": contrast,
                "feedback_good": good, "feedback_bad": bad,
                "requests_24h": total_24h,
            }));
        }

        Ok(json!({ "nodes": nodes, "links": links }))
    }

    /// Full-text search over cached prompts — returns lightweight summaries for the graph sidebar.
    pub async fn search_entries_for_graph(
        &self,
        query:         &str,
        domain_filter: Option<&str>,
        limit:         i64,
    ) -> Result<Vec<serde_json::Value>> {
        use serde_json::json;
        let domain_flag = domain_filter.unwrap_or("all");
        let rows = if !query.trim().is_empty() {
            let clean: String = query.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '_')
                .collect::<String>();
            let fts_term = format!("{}*", clean.trim());

            let result = sqlx::query(
                "SELECT ce.id, ce.domain, ce.intent, ce.complexity,
                        SUBSTR(ce.prompt_text, 1, 200) AS preview,
                        ce.model_used, ce.confidence, ce.hit_count, ce.created_at
                 FROM cache_entries_fts fts
                 JOIN cache_entries ce ON ce.id = fts.id
                 WHERE fts MATCH ?
                   AND (? = 'all' OR ce.domain = ?)
                 ORDER BY ce.hit_count DESC, ce.created_at DESC LIMIT ?"
            )
            .bind(&fts_term)
            .bind(domain_flag)
            .bind(domain_filter.unwrap_or(""))
            .bind(limit)
            .fetch_all(&self.pool)
            .await;

            match result {
                Ok(rows) => rows,
                Err(_) => {
                    let pat = format!("%{}%", query);
                    sqlx::query(
                        "SELECT id, domain, intent, complexity,
                                SUBSTR(prompt_text, 1, 200) AS preview,
                                model_used, confidence, hit_count, created_at
                         FROM cache_entries
                         WHERE prompt_text LIKE ?
                           AND (? = 'all' OR domain = ?)
                         ORDER BY hit_count DESC, created_at DESC LIMIT ?"
                    )
                    .bind(&pat)
                    .bind(domain_flag)
                    .bind(domain_filter.unwrap_or(""))
                    .bind(limit)
                    .fetch_all(&self.pool).await?
                }
            }
        } else {
            sqlx::query(
                "SELECT id, domain, intent, complexity,
                        SUBSTR(prompt_text, 1, 200) AS preview,
                        model_used, confidence, hit_count, created_at
                 FROM cache_entries
                 WHERE (? = 'all' OR domain = ?)
                 ORDER BY hit_count DESC, created_at DESC LIMIT ?"
            )
            .bind(domain_flag)
            .bind(domain_filter.unwrap_or(""))
            .bind(limit)
            .fetch_all(&self.pool).await?
        };
        Ok(rows.iter().map(|r| json!({
            "id":         r.get::<String, _>("id"),
            "domain":     r.get::<String, _>("domain"),
            "intent":     r.get::<String, _>("intent"),
            "complexity": r.get::<f64,    _>("complexity"),
            "preview":    r.get::<String, _>("preview"),
            "model_used": r.get::<String, _>("model_used"),
            "confidence": r.get::<Option<f64>, _>("confidence"),
            "hit_count":  r.get::<i64,    _>("hit_count"),
            "created_at": r.get::<i64,    _>("created_at"),
        })).collect())
    }

    /// Reconstruct the routing trace for a specific cache entry from DB state.
    /// Scores are estimates derived from stored metadata; current system state is used
    /// for layer 2/3 info since per-request trace storage is not yet implemented.
    pub async fn entry_trace(&self, cache_id: &str, novelty_threshold: f64) -> Result<serde_json::Value> {
        use serde_json::json;
        let row = sqlx::query(
            "SELECT id, domain, intent, complexity,
                    SUBSTR(prompt_text, 1, 300) AS preview,
                    model_used, confidence, hit_count, created_at, shared
             FROM cache_entries WHERE id = ?"
        ).bind(cache_id).fetch_optional(&self.pool).await?
         .ok_or_else(|| anyhow::anyhow!("entry not found: {}", cache_id))?;

        let domain:     String       = row.get("domain");
        let intent:     String       = row.get("intent");
        let complexity: f64          = row.get("complexity");
        let model_used: String       = row.get("model_used");
        let confidence: Option<f64>  = row.get("confidence");
        let hit_count:  i64          = row.get("hit_count");

        let doc = sqlx::query(
            "SELECT version, length(content) AS chars, entry_count, updated_at
             FROM domain_knowledge WHERE domain = ?"
        ).bind(&domain).fetch_optional(&self.pool).await?;

        let thresh = sqlx::query(
            "SELECT novelty_override, escalation_rate, sample_count
             FROM routing_thresholds WHERE domain = ? AND intent = ?"
        ).bind(&domain).bind(&intent).fetch_optional(&self.pool).await?;

        let contrast_rows = sqlx::query(
            "SELECT SUBSTR(prompt_text, 1, 120) AS preview, local_confidence, created_at
             FROM escalation_pairs WHERE domain = ?
             ORDER BY created_at DESC LIMIT 5"
        ).bind(&domain).fetch_all(&self.pool).await.unwrap_or_default();

        let since = Utc::now().timestamp() - 86400;
        let stats = sqlx::query(
            "SELECT COUNT(*) AS total,
                    SUM(CASE WHEN decision='api'           THEN 1 ELSE 0 END) AS api_n,
                    SUM(CASE WHEN decision='local'         THEN 1 ELSE 0 END) AS local_n,
                    SUM(CASE WHEN decision IN ('exact_cache','semantic_cache') THEN 1 ELSE 0 END) AS cache_n,
                    AVG(latency_ms) AS avg_lat
             FROM routing_log WHERE domain = ? AND intent = ? AND created_at >= ?"
        ).bind(&domain).bind(&intent).bind(since)
         .fetch_optional(&self.pool).await?;

        let eff_threshold = thresh.as_ref()
            .map(|t| t.get::<f64,_>("novelty_override"))
            .unwrap_or(novelty_threshold);

        // Prefer actual gate scores persisted at routing time; fall back to estimates.
        let scores_row = sqlx::query(
            "SELECT scores_json FROM routing_log
             WHERE domain = ? AND intent = ? AND scores_json IS NOT NULL
             ORDER BY created_at DESC LIMIT 1"
        ).bind(&domain).bind(&intent)
         .fetch_optional(&self.pool).await?;

        let (novelty_val, complexity_val, consequence_val, scores_source) =
            if let Some(ref row) = scores_row {
                let json_str: String = row.get("scores_json");
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    let n = v["novelty"].as_f64().unwrap_or(0.0);
                    let c = v["complexity"].as_f64().unwrap_or(complexity);
                    let q = v["consequence"].as_f64().unwrap_or(0.0);
                    (n, c, q, "recorded")
                } else {
                    let est = match hit_count.max(0) {
                        0 => 0.80_f64, 1 => 0.50, 2..=4 => 0.35, 5..=19 => 0.20, _ => 0.05,
                    };
                    (est, complexity, domain::consequence_score(&domain, &intent), "estimated")
                }
            } else {
                let est = match hit_count.max(0) {
                    0 => 0.80_f64, 1 => 0.50, 2..=4 => 0.35, 5..=19 => 0.20, _ => 0.05,
                };
                (est, complexity, domain::consequence_score(&domain, &intent), "estimated")
            };

        // Count similar entries in same domain (rough proxy for L1 candidate pool)
        let similar_count: i64 = sqlx::query(
            "SELECT COUNT(*) AS n FROM cache_entries WHERE domain = ? AND id != ?"
        ).bind(&domain).bind(cache_id)
         .fetch_one(&self.pool).await.map(|r| r.get("n")).unwrap_or(0);

        Ok(json!({
            "entry": {
                "id": row.get::<String,_>("id"),
                "domain": domain, "intent": intent,
                "complexity": complexity,
                "preview": row.get::<String,_>("preview"),
                "model_used": model_used,
                "confidence": confidence,
                "hit_count": hit_count,
                "created_at": row.get::<i64,_>("created_at"),
            },
            "knowledge_doc": doc.as_ref().map(|d| json!({
                "version":     d.get::<i64,_>("version"),
                "doc_chars":   d.get::<i64,_>("chars"),
                "entry_count": d.get::<i64,_>("entry_count"),
            })),
            "threshold": thresh.as_ref().map(|t| json!({
                "override":       t.get::<f64,_>("novelty_override"),
                "base":           novelty_threshold,
                "escalation_rate": t.get::<f64,_>("escalation_rate"),
                "sample_count":   t.get::<i64,_>("sample_count"),
                "adapted":        (t.get::<f64,_>("novelty_override") - novelty_threshold).abs() > 1e-9,
            })),
            "contrast_pairs": contrast_rows.iter().map(|r| json!({
                "preview":    r.get::<String,_>("preview"),
                "confidence": r.get::<Option<f64>,_>("local_confidence"),
            })).collect::<Vec<_>>(),
            "routing_stats": stats.as_ref().map(|s| json!({
                "total_24h": s.get::<i64,_>("total"),
                "api_24h":   s.get::<i64,_>("api_n"),
                "local_24h": s.get::<i64,_>("local_n"),
                "cache_24h": s.get::<i64,_>("cache_n"),
                "avg_latency_ms": s.get::<f64,_>("avg_lat"),
            })),
            "scores": {
                "novelty":      novelty_val,
                "complexity":   complexity_val,
                "consequence":  consequence_val,
                "threshold":    eff_threshold,
                "adapted":      thresh.as_ref().map(|t| (t.get::<f64,_>("novelty_override") - novelty_threshold).abs() > 1e-9).unwrap_or(false),
                "source":       scores_source,
            },
            "similar_count": similar_count,
        }))
    }

    /// Aggregate per-domain learning state for the /api/learning/brain endpoint.
    /// Returns a snapshot of how each layer has grown across all known domains.
    pub async fn brain_snapshot(&self, window_secs: i64) -> Result<serde_json::Value> {
        use serde_json::json;
        use std::collections::HashSet;
        let since = Utc::now().timestamp() - window_secs;

        let doc_rows = sqlx::query(
            "SELECT domain, entry_count, version, length(content) AS doc_chars, updated_at
             FROM domain_knowledge ORDER BY domain"
        ).fetch_all(&self.pool).await.unwrap_or_default();

        let thresh_rows = sqlx::query(
            "SELECT domain, intent, novelty_override, escalation_rate, sample_count
             FROM routing_thresholds ORDER BY domain, intent"
        ).fetch_all(&self.pool).await.unwrap_or_default();

        let route_rows = sqlx::query(
            "SELECT domain,
                    COUNT(*) AS total,
                    SUM(CASE WHEN decision = 'api' THEN 1 ELSE 0 END) AS api_calls,
                    AVG(latency_ms) AS avg_latency
             FROM routing_log WHERE created_at >= ? AND domain IS NOT NULL
             GROUP BY domain"
        ).bind(since).fetch_all(&self.pool).await.unwrap_or_default();

        let contrast_rows = sqlx::query(
            "SELECT domain, COUNT(*) AS count FROM escalation_pairs GROUP BY domain"
        ).fetch_all(&self.pool).await.unwrap_or_default();

        let feedback_rows = sqlx::query(
            "SELECT domain, signal, COUNT(*) AS count
             FROM response_feedback WHERE created_at >= ?
             GROUP BY domain, signal"
        ).bind(since).fetch_all(&self.pool).await.unwrap_or_default();

        let mut domains: HashSet<String> = HashSet::new();
        for r in &doc_rows     { domains.insert(r.get("domain")); }
        for r in &thresh_rows  { domains.insert(r.get("domain")); }
        for r in &route_rows   { if let Some(d) = r.get::<Option<String>, _>("domain") { domains.insert(d); } }
        for r in &contrast_rows { domains.insert(r.get("domain")); }
        for r in &feedback_rows { domains.insert(r.get("domain")); }

        let mut sorted: Vec<String> = domains.into_iter().collect();
        sorted.sort();

        let mut domain_list: Vec<serde_json::Value> = Vec::new();
        for domain in &sorted {
            let doc_info = doc_rows.iter().find(|r| r.get::<String, _>("domain") == *domain);
            let thresholds: Vec<serde_json::Value> = thresh_rows.iter()
                .filter(|r| r.get::<String, _>("domain") == *domain)
                .map(|r| json!({
                    "intent":          r.get::<String, _>("intent"),
                    "override":        r.get::<f64, _>("novelty_override"),
                    "escalation_rate": r.get::<f64, _>("escalation_rate"),
                    "sample_count":    r.get::<i64, _>("sample_count"),
                }))
                .collect();
            let route_info = route_rows.iter()
                .find(|r| r.get::<Option<String>, _>("domain").as_deref() == Some(domain.as_str()));
            let contrast_count: i64 = contrast_rows.iter()
                .find(|r| r.get::<String, _>("domain") == *domain)
                .map(|r| r.get("count"))
                .unwrap_or(0);
            let good_count: i64 = feedback_rows.iter()
                .filter(|r| r.get::<String, _>("domain") == *domain && r.get::<String, _>("signal") == "good")
                .map(|r| r.get::<i64, _>("count")).sum();
            let bad_count: i64 = feedback_rows.iter()
                .filter(|r| r.get::<String, _>("domain") == *domain && r.get::<String, _>("signal") == "bad")
                .map(|r| r.get::<i64, _>("count")).sum();

            domain_list.push(json!({
                "domain": domain,
                "knowledge_doc": doc_info.map(|r| json!({
                    "version":     r.get::<i64, _>("version"),
                    "entry_count": r.get::<i64, _>("entry_count"),
                    "doc_chars":   r.get::<i64, _>("doc_chars"),
                    "updated_at":  r.get::<i64, _>("updated_at"),
                })),
                "thresholds": thresholds,
                "routing": route_info.map(|r| {
                    let total: i64    = r.get("total");
                    let api_calls: i64 = r.get("api_calls");
                    json!({
                        "total_requests":  total,
                        "api_calls":       api_calls,
                        "escalation_rate": if total > 0 { api_calls as f64 / total as f64 } else { 0.0 },
                        "avg_latency_ms":  r.get::<f64, _>("avg_latency"),
                    })
                }),
                "contrast_pairs": contrast_count,
                "feedback": { "good": good_count, "bad": bad_count },
            }));
        }

        let total_entries: i64 = sqlx::query("SELECT COUNT(*) AS n FROM cache_entries")
            .fetch_one(&self.pool).await.map(|r| r.get("n")).unwrap_or(0);
        let total_contrasts: i64 = sqlx::query("SELECT COUNT(*) AS n FROM escalation_pairs")
            .fetch_one(&self.pool).await.map(|r| r.get("n")).unwrap_or(0);
        let knowledge_domains: i64 = sqlx::query("SELECT COUNT(*) AS n FROM domain_knowledge")
            .fetch_one(&self.pool).await.map(|r| r.get("n")).unwrap_or(0);

        Ok(json!({
            "snapshot_window_secs": window_secs,
            "total_cache_entries":  total_entries,
            "total_contrast_pairs": total_contrasts,
            "knowledge_domains":    knowledge_domains,
            "layers_active": ["L1_fewshot", "L2_distillation", "L3_adaptive_threshold", "L4_feedback", "L5_contrastive"],
            "domains": domain_list,
        }))
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Truncate `s` to at most `max_bytes` bytes, aligned to a valid UTF-8 char boundary.
fn truncate_chars(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes { return s; }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) { end -= 1; }
    &s[..end]
}

fn normalize_prompt(p: &str) -> String {
    p.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn row_to_entry(row: sqlx::sqlite::SqliteRow) -> CacheEntry {
    CacheEntry {
        id:          row.get("id"),
        domain:      row.get("domain"),
        intent:      row.get("intent"),
        complexity:  row.get("complexity"),
        prompt_text: row.get("prompt_text"),
        response:    row.get("response"),
        model_used:  row.get("model_used"),
        confidence:  row.get("confidence"),
        created_at:  row.get("created_at"),
        expires_at:  row.get("expires_at"),
        hit_count:   row.get("hit_count"),
        last_hit_at: row.try_get("last_hit_at").ok(),
        node_id:     row.get("node_id"),
        shared:      row.get::<i32, _>("shared") != 0,
        pinned:      row.get::<i32, _>("pinned") != 0,
    }
}

fn row_to_entry_partial(row: &sqlx::sqlite::SqliteRow) -> CacheEntry {
    CacheEntry {
        id:          row.get("id"),
        domain:      row.get("domain"),
        intent:      row.get("intent"),
        complexity:  row.get("complexity"),
        prompt_text: row.get("prompt_text"),
        response:    row.get("response"),
        model_used:  row.get("model_used"),
        confidence:  row.get("confidence"),
        created_at:  row.get("created_at"),
        expires_at:  row.get("expires_at"),
        hit_count:   row.get("hit_count"),
        last_hit_at: row.try_get("last_hit_at").ok(),
        node_id:     row.get("node_id"),
        shared:      row.get::<i32, _>("shared") != 0,
        pinned:      row.get::<i32, _>("pinned") != 0,
    }
}

pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}


pub fn decode_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 { return None; }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot:   f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let mag_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 { return 0.0; }
    dot / (mag_a * mag_b)
}

/// Extract text content from a stored response JSON blob without pulling in the
/// full backend types — used by the cache layer for distillation source prep.
fn extract_answer_from_response_json(response_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(response_json).ok()?;
    let text = v.get("content")?
        .as_array()?
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|b| b.get("text")?.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() { None } else { Some(text) }
}
