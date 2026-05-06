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
    /// SHA256(normalized_prompt) — the DHT key for federation
    pub id:          String,
    pub domain:      String,
    pub intent:      String,
    pub complexity:  f64,
    pub prompt_text: String,
    pub response:    String, // full JSON blob
    pub model_used:  String,
    pub confidence:  Option<f64>,
    pub created_at:  i64,
    pub expires_at:  Option<i64>,
    pub hit_count:   i64,
    pub node_id:     String,
    pub shared:      bool,
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

// ── Store ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct CacheStore {
    pool:    SqlitePool,
    node_id: String,
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

        let store = CacheStore { pool, node_id: node_id.to_string() };
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
                shared       INTEGER NOT NULL DEFAULT 0
            )
        "#).execute(&self.pool).await?;

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

        Ok(())
    }

    // ── Key derivation ───────────────────────────────────────────────────

    /// Content-addressed key: SHA256 of lowercased, whitespace-normalized prompt.
    /// This is also the natural DHT key for P2P federation.
    pub fn content_key(prompt: &str) -> String {
        let normalized = normalize_prompt(prompt);
        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        hex::encode(hasher.finalize())
    }

    // ── Lookup ───────────────────────────────────────────────────────────

    /// Exact-match cache lookup by SHA256 key.
    pub async fn lookup_exact(&self, prompt: &str) -> Result<Option<CacheEntry>> {
        let key = Self::content_key(prompt);
        let now = Utc::now().timestamp();

        let row = sqlx::query(
            "SELECT id, domain, intent, complexity, prompt_text, response, model_used,
                    confidence, created_at, expires_at, hit_count, node_id, shared
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

        // Load all live embeddings in this domain — intentionally bounded by domain
        // to keep the in-memory set manageable.
        let rows = sqlx::query(
            "SELECT ce.id, ce.domain, ce.intent, ce.complexity, ce.prompt_text, ce.response,
                    ce.model_used, ce.confidence, ce.created_at, ce.expires_at, ce.hit_count,
                    ce.node_id, ce.shared, emb.embedding, emb.model
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
                    let entry = row_to_entry_partial(&row);
                    Some((entry, sim))
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

    // ── Store ────────────────────────────────────────────────────────────

    pub async fn store(
        &self,
        shape: &ShapeKey,
        prompt: &str,
        response: &str,
        model_used: &str,
        confidence: Option<f64>,
        ttl_secs: Option<u64>,
        shared: bool,
    ) -> Result<String> {
        let id = Self::content_key(prompt);
        let now = Utc::now().timestamp();
        let expires_at: Option<i64> = ttl_secs.map(|t| now + t as i64);

        sqlx::query(
            "INSERT INTO cache_entries
                (id, domain, intent, complexity, prompt_text, response, model_used,
                 confidence, created_at, expires_at, hit_count, node_id, shared)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
             ON CONFLICT(id) DO UPDATE SET
                response   = excluded.response,
                model_used = excluded.model_used,
                confidence = excluded.confidence,
                expires_at = excluded.expires_at,
                shared     = excluded.shared"
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
        .execute(&self.pool)
        .await?;

        Ok(id)
    }

    /// Store an embedding vector alongside a cache entry.
    pub async fn store_embedding(
        &self,
        cache_id: &str,
        embedding: &[f32],
        model: &str,
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

    // ── Federation helpers ───────────────────────────────────────────────

    /// Look up a cache entry by its raw SHA256 hash — used by federation peers.
    pub async fn lookup_by_hash(&self, hash: &str) -> Result<Option<CacheEntry>> {
        let now = Utc::now().timestamp();
        let row = sqlx::query(
            "SELECT id, domain, intent, complexity, prompt_text, response, model_used,
                    confidence, created_at, expires_at, hit_count, node_id, shared
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

        Ok(CacheStats {
            total_entries:  total,
            total_hits:     hits,
            shared_entries: shared,
            db_size_bytes:  0, // filled by portal if needed
        })
    }

    pub async fn log_routing(
        &self,
        shape_key: &str,
        decision: &str,
        backend: &str,
        latency_ms: i64,
        tokens_in: Option<i64>,
        tokens_out: Option<i64>,
        saved_usd: Option<f64>,
    ) -> Result<()> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO routing_log (id, shape_key, decision, backend, latency_ms,
                                      tokens_in, tokens_out, saved_usd, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&id)
        .bind(shape_key)
        .bind(decision)
        .bind(backend)
        .bind(latency_ms)
        .bind(tokens_in)
        .bind(tokens_out)
        .bind(saved_usd)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Purge all cache entries originating from a specific node.
    /// Called during federation node eviction.
    pub async fn purge_by_node_id(&self, node_id: &str) -> Result<u64> {
        let r = sqlx::query("DELETE FROM cache_entries WHERE node_id = ?")
            .bind(node_id)
            .execute(&self.pool)
            .await?;
        Ok(r.rows_affected())
    }

    /// Evict expired entries.
    pub async fn evict_expired(&self) -> Result<u64> {
        let now = Utc::now().timestamp();
        let r = sqlx::query("DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?")
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
    }
}

pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn decode_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let mag_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}
