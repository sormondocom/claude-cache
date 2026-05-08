use anyhow::Result;
use chrono::Utc;
use sha2::{Digest, Sha256};
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions},
    Row,
};
use std::str::FromStr;

use crate::domain::ShapeKey;

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
        let _ = sqlx::query("ALTER TABLE cache_entries ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN miss_reason TEXT")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN domain TEXT")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE routing_log ADD COLUMN intent TEXT")
            .execute(&self.pool).await;

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
                    confidence, created_at, expires_at, hit_count, node_id, shared, pinned
             FROM cache_entries
             WHERE id = ? AND (expires_at IS NULL OR expires_at > ?)"
        )
        .bind(&key)
        .bind(now)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            self.bump_hit(&key).await?;
            return Ok(Some(row_to_entry(row)));
        }
        Ok(None)
    }

    /// Semantic lookup: returns entries in the same domain whose stored embedding
    /// is within `sim_threshold` cosine similarity of `query_embedding`.
    pub async fn lookup_semantic(
        &self,
        domain: &str,
        query_embedding: &[f32],
        sim_threshold: f64,
        limit: usize,
    ) -> Result<Vec<(CacheEntry, f64)>> {
        let now = Utc::now().timestamp();

        let rows = sqlx::query(
            "SELECT ce.id, ce.domain, ce.intent, ce.complexity, ce.prompt_text, ce.response,
                    ce.model_used, ce.confidence, ce.created_at, ce.expires_at, ce.hit_count,
                    ce.node_id, ce.shared, ce.pinned, emb.embedding, emb.model
             FROM cache_entries ce
             JOIN cache_embeddings emb ON emb.cache_id = ce.id
             WHERE ce.domain = ? AND (ce.expires_at IS NULL OR ce.expires_at > ?)
             ORDER BY ce.hit_count DESC
             LIMIT 500"
        )
        .bind(domain)
        .bind(now)
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

        if let Some((entry, _)) = results.first() {
            self.bump_hit(&entry.id).await?;
        }

        Ok(results)
    }

    /// Returns the highest cosine similarity found for any live entry in this domain,
    /// regardless of threshold.  Does NOT bump hit counts — used only by the routing
    /// gate to distinguish "near miss" from "truly never seen."
    pub async fn best_semantic_sim(
        &self,
        domain: &str,
        query_embedding: &[f32],
    ) -> Result<Option<f64>> {
        let now = Utc::now().timestamp();
        let rows = sqlx::query(
            "SELECT emb.embedding
             FROM cache_entries ce
             JOIN cache_embeddings emb ON emb.cache_id = ce.id
             WHERE ce.domain = ? AND (ce.expires_at IS NULL OR ce.expires_at > ?)
             LIMIT 500"
        )
        .bind(domain)
        .bind(now)
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
    /// Returns summaries ordered by hit_count DESC.
    pub async fn search_entries(
        &self,
        query:  Option<&str>,
        domain: Option<&str>,
        limit:  i64,
    ) -> Result<Vec<CacheEntrySummary>> {
        let now  = Utc::now().timestamp();
        // "all" sentinel skips the domain equality check; "%" matches everything.
        let domain_flag   = domain.unwrap_or("all");
        let like_pattern  = query
            .map(|q| format!("%{}%", q.to_lowercase()))
            .unwrap_or_else(|| "%".to_string());

        let rows = sqlx::query(
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
        .await?;

        Ok(rows.into_iter().map(|row| {
            let text: String = row.get("prompt_text");
            CacheEntrySummary {
                id:             row.get("id"),
                domain:         row.get("domain"),
                intent:         row.get("intent"),
                prompt_preview: if text.len() > 200 { format!("{}…", &text[..200]) } else { text },
                model_used:     row.get("model_used"),
                hit_count:      row.get("hit_count"),
                pinned:         row.get::<i32, _>("pinned") != 0,
                created_at:     row.get("created_at"),
                expires_at:     row.get("expires_at"),
            }
        }).collect())
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

    /// Return a page of shared hashes — used to gossip our inventory to peers.
    pub async fn list_shared_hashes(&self, limit: i64, offset: i64) -> Result<Vec<String>> {
        let rows = sqlx::query(
            "SELECT id FROM cache_entries WHERE shared = 1 ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
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
        for _ in 0..200 {
            let row = sqlx::query(
                "SELECT (page_count - freelist_count) * page_size AS live \
                 FROM pragma_page_count(), pragma_freelist_count(), pragma_page_size()"
            ).fetch_one(&self.pool).await?;
            let live_bytes: i64 = row.get("live");
            if (live_bytes as u64) <= max_bytes { break; }

            let r = sqlx::query(
                "DELETE FROM cache_entries WHERE id IN (
                     SELECT id FROM cache_entries
                     WHERE pinned = 0
                     ORDER BY COALESCE(last_hit_at, created_at) ASC
                     LIMIT 100
                 )"
            ).execute(&self.pool).await?;
            let deleted = r.rows_affected();
            if deleted == 0 { break; }
            total_evicted += deleted;
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
    ) -> Result<()> {
        let id  = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO routing_log (id, shape_key, domain, intent, decision, backend,
                                      latency_ms, tokens_in, tokens_out, saved_usd,
                                      miss_reason, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
                    saved_usd, miss_reason, created_at
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
            "UPDATE cache_entries SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?"
        )
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

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
