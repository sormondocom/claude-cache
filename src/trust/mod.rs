/// Trust store — PGP-style web-of-trust for federation nodes.
///
/// Trust states:
///   Untrusted — node is known but not verified. No cache entries accepted.
///   Trusted   — manually promoted, or promoted by a trusted head node's
///               countersignature. Cache entries are accepted.
///   Evicted   — permanently blocked. All their cached entries are purged on
///               eviction. Signed revocation can be gossiped to peers.
///
/// The head node pattern: when a node is promoted with `head_node = true`,
/// it can counter-sign other nodes' public keys.  A counter-signature from a
/// trusted head node is enough to auto-promote a new peer to Trusted.

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions},
    Row,
};
use std::str::FromStr;
use tracing::{info, warn};

use crate::cache::CacheStore;
use crate::identity::RemoteKey;

// ── Health types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct PeerHealth {
    pub node_id:          String,
    pub url:              String,
    pub is_reachable:     bool,
    /// Latency of the most recent successful check (ms).
    pub latency_ms:       Option<i64>,
    /// Exponential moving average latency across all successful checks (ms).
    pub avg_latency_ms:   Option<f64>,
    pub last_checked:     Option<i64>,
    pub last_success:     Option<i64>,
    pub consecutive_fail: u32,
    pub check_count:      i64,
}

// ── State types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum NodeTrustState {
    Untrusted,
    Trusted   { signed_by: String },
    Evicted   { reason: String, at: i64 },
}

impl NodeTrustState {
    pub fn is_trusted(&self) -> bool {
        matches!(self, NodeTrustState::Trusted { .. })
    }
    pub fn is_evicted(&self) -> bool {
        matches!(self, NodeTrustState::Evicted { .. })
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeTrustState::Untrusted       => "untrusted",
            NodeTrustState::Trusted { .. }  => "trusted",
            NodeTrustState::Evicted { .. }  => "evicted",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRecord {
    pub node_id:       String,
    pub public_key_hex: String,
    pub url:           String,
    pub is_head:       bool,
    pub trust:         NodeTrustState,
    pub first_seen:    i64,
    pub last_seen:     i64,
}

// ── Store ────────────────────────────────────────────────────────────────────

pub struct TrustStore {
    pool: SqlitePool,
}

impl TrustStore {
    pub async fn open(db_path: &str, _own_node_id: &str) -> Result<Self> {
        let opts = SqliteConnectOptions::from_str(&format!("sqlite:{db_path}"))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(opts)
            .await?;

        let store = TrustStore { pool };
        store.migrate().await?;
        Ok(store)
    }

    async fn migrate(&self) -> Result<()> {
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS node_records (
                node_id        TEXT    PRIMARY KEY,
                public_key_hex TEXT    NOT NULL,
                url            TEXT    NOT NULL DEFAULT '',
                is_head        INTEGER NOT NULL DEFAULT 0,
                trust_state    TEXT    NOT NULL DEFAULT 'untrusted',
                signed_by      TEXT,
                evict_reason   TEXT,
                evict_at       INTEGER,
                first_seen     INTEGER NOT NULL,
                last_seen      INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS trust_events (
                id          TEXT    PRIMARY KEY,
                node_id     TEXT    NOT NULL,
                event       TEXT    NOT NULL,
                actor       TEXT    NOT NULL,
                reason      TEXT,
                created_at  INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS revocations (
                node_id     TEXT    PRIMARY KEY,
                revoked_by  TEXT    NOT NULL,
                reason      TEXT    NOT NULL,
                signature   TEXT    NOT NULL,
                revoked_at  INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS peer_health (
                node_id          TEXT    PRIMARY KEY,
                url              TEXT    NOT NULL DEFAULT '',
                is_reachable     INTEGER NOT NULL DEFAULT 1,
                latency_ms       INTEGER,
                avg_latency_ms   REAL,
                last_checked     INTEGER,
                last_success     INTEGER,
                consecutive_fail INTEGER NOT NULL DEFAULT 0,
                consecutive_ok   INTEGER NOT NULL DEFAULT 0,
                check_count      INTEGER NOT NULL DEFAULT 0
            )
        "#).execute(&self.pool).await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_trust_state ON node_records(trust_state)")
            .execute(&self.pool).await?;

        Ok(())
    }

    // ── Registration ────────────────────────────────────────────────────────

    /// Register a new node from its announce payload.  The signature must have
    /// been verified BEFORE calling this — this method only manages state.
    /// Returns the trust state after registration.
    pub async fn register(&self, node_id: &str, public_key_hex: &str, url: &str) -> Result<NodeTrustState> {
        let now = Utc::now().timestamp();

        // Insert or update last_seen; do NOT change trust_state if already set.
        sqlx::query(r#"
            INSERT INTO node_records (node_id, public_key_hex, url, trust_state, first_seen, last_seen)
            VALUES (?, ?, ?, 'untrusted', ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                url            = excluded.url,
                public_key_hex = CASE WHEN node_records.public_key_hex = ''
                                      THEN excluded.public_key_hex
                                      ELSE node_records.public_key_hex END,
                last_seen      = excluded.last_seen
        "#)
        .bind(node_id)
        .bind(public_key_hex)
        .bind(url)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;

        self.get_state(node_id).await
    }

    /// Register a peer that was declared in config and auto-trust it.
    /// Config-specified peers are explicitly trusted by the operator — this is
    /// the equivalent of SSH known_hosts.  If the peer record already exists and
    /// is trusted/evicted, the trust state is not downgraded.
    pub async fn register_config_peer(
        &self,
        node_id:        &str,
        public_key_hex: &str,
        url:            &str,
        is_head:        bool,
    ) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query(r#"
            INSERT INTO node_records
                (node_id, public_key_hex, url, is_head, trust_state, signed_by, first_seen, last_seen)
            VALUES (?, ?, ?, ?, 'trusted', 'config', ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                url            = excluded.url,
                is_head        = excluded.is_head,
                public_key_hex = CASE WHEN node_records.public_key_hex = ''
                                      THEN excluded.public_key_hex
                                      ELSE node_records.public_key_hex END,
                last_seen      = excluded.last_seen
        "#)
        .bind(node_id)
        .bind(public_key_hex)
        .bind(url)
        .bind(is_head as i32)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;
        self.log_event(node_id, "config-trusted", "config", None).await;
        Ok(())
    }

    // ── Queries ─────────────────────────────────────────────────────────────

    pub async fn get_state(&self, node_id: &str) -> Result<NodeTrustState> {
        let row = sqlx::query(
            "SELECT trust_state, signed_by, evict_reason, evict_at FROM node_records WHERE node_id = ?"
        )
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = row else {
            return Ok(NodeTrustState::Untrusted);
        };

        let state: String = row.get("trust_state");
        Ok(match state.as_str() {
            "trusted" => NodeTrustState::Trusted {
                signed_by: row.get::<Option<String>, _>("signed_by")
                    .unwrap_or_else(|| "manual".into()),
            },
            "evicted" => NodeTrustState::Evicted {
                reason: row.get::<Option<String>, _>("evict_reason")
                    .unwrap_or_default(),
                at: row.get::<Option<i64>, _>("evict_at")
                    .unwrap_or(0),
            },
            _ => NodeTrustState::Untrusted,
        })
    }

    pub async fn is_trusted(&self, node_id: &str) -> bool {
        self.get_state(node_id).await
            .map(|s| s.is_trusted())
            .unwrap_or(false)
    }

    pub async fn is_evicted(&self, node_id: &str) -> bool {
        self.get_state(node_id).await
            .map(|s| s.is_evicted())
            .unwrap_or(false)
    }

    pub async fn get_record(&self, node_id: &str) -> Result<Option<NodeRecord>> {
        let row = sqlx::query(
            "SELECT node_id, public_key_hex, url, is_head, trust_state, signed_by,
                    evict_reason, evict_at, first_seen, last_seen
             FROM node_records WHERE node_id = ?"
        )
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(row_to_record))
    }

    pub async fn get_public_key(&self, node_id: &str) -> Result<Option<RemoteKey>> {
        let row = sqlx::query("SELECT public_key_hex FROM node_records WHERE node_id = ?")
            .bind(node_id)
            .fetch_optional(&self.pool)
            .await?;
        match row {
            Some(r) => {
                let hex: String = r.get("public_key_hex");
                Ok(Some(RemoteKey::from_hex(&hex)?))
            }
            None => Ok(None),
        }
    }

    /// Apply a revocation received from a peer.
    /// Verifies that `revoked_by` is a trusted node with a known key before applying.
    /// Returns `true` if the revocation was newly applied, `false` if already known or invalid.
    pub async fn apply_incoming_revocation(
        &self,
        rev: &RevocationRecord,
        cache: &CacheStore,
    ) -> Result<bool> {
        // Already handled — idempotent
        if self.is_evicted(&rev.node_id).await {
            return Ok(false);
        }

        // The node that issued the revocation must be trusted by us
        if !self.is_trusted(&rev.revoked_by).await {
            warn!(
                "revocation from untrusted node {} ignored (target: {})",
                &rev.revoked_by[..16.min(rev.revoked_by.len())],
                &rev.node_id[..16.min(rev.node_id.len())]
            );
            return Ok(false);
        }

        // Verify the signature using the revoker's stored public key
        let Some(key) = self.get_public_key(&rev.revoked_by).await? else {
            warn!("no public key on record for revoker {}", &rev.revoked_by[..16.min(rev.revoked_by.len())]);
            return Ok(false);
        };

        let msg = crate::identity::revocation_message(&rev.node_id, &rev.reason);
        if let Err(e) = key.verify(&msg, &rev.signature) {
            warn!("revocation signature invalid: {e}");
            return Ok(false);
        }

        // Apply — use the original revoker as actor, preserve their signature
        self.evict(&rev.node_id, &rev.reason, &rev.revoked_by, &rev.signature, cache).await?;
        Ok(true)
    }

    pub async fn list_all(&self) -> Result<Vec<NodeRecord>> {
        let rows = sqlx::query(
            "SELECT node_id, public_key_hex, url, is_head, trust_state, signed_by,
                    evict_reason, evict_at, first_seen, last_seen
             FROM node_records ORDER BY last_seen DESC"
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(row_to_record).collect())
    }

    pub async fn list_trusted(&self) -> Result<Vec<NodeRecord>> {
        let rows = sqlx::query(
            "SELECT node_id, public_key_hex, url, is_head, trust_state, signed_by,
                    evict_reason, evict_at, first_seen, last_seen
             FROM node_records WHERE trust_state = 'trusted' ORDER BY last_seen DESC"
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(row_to_record).collect())
    }

    // ── Mutations ───────────────────────────────────────────────────────────

    /// Manually promote a node to Trusted.  `actor` is the operator's node_id.
    pub async fn promote(&self, node_id: &str, actor: &str, is_head: bool) -> Result<()> {
        sqlx::query(
            "UPDATE node_records SET trust_state = 'trusted', signed_by = ?, is_head = ?
             WHERE node_id = ?"
        )
        .bind(actor)
        .bind(is_head as i32)
        .bind(node_id)
        .execute(&self.pool)
        .await?;

        self.log_event(node_id, "promoted", actor, None).await;
        info!("node {node_id} promoted to trusted by {actor}");
        Ok(())
    }

    /// Head-node counter-signature auto-promotion.
    /// If `signed_by_node_id` is a trusted head node, the target is auto-promoted.
    pub async fn auto_promote_if_head_signed(
        &self,
        target_node_id: &str,
        signed_by_node_id: &str,
    ) -> Result<bool> {
        let row = sqlx::query(
            "SELECT is_head, trust_state FROM node_records WHERE node_id = ?"
        )
        .bind(signed_by_node_id)
        .fetch_optional(&self.pool)
        .await?;

        let is_head = row
            .as_ref()
            .and_then(|r| Some(r.get::<i32, _>("is_head") != 0))
            .unwrap_or(false);
        let is_trusted = row
            .as_ref()
            .and_then(|r| {
                let s: String = r.get("trust_state");
                Some(s == "trusted")
            })
            .unwrap_or(false);

        if is_head && is_trusted {
            self.promote(target_node_id, signed_by_node_id, false).await?;
            info!("node {target_node_id} auto-promoted via head-node {signed_by_node_id}");
            return Ok(true);
        }
        Ok(false)
    }

    /// Evict a node: update state, write a signed revocation, purge its cache entries.
    pub async fn evict(
        &self,
        node_id:    &str,
        reason:     &str,
        actor:      &str,
        revocation_sig: &str,
        cache:      &CacheStore,
    ) -> Result<()> {
        let now = Utc::now().timestamp();

        sqlx::query(
            "UPDATE node_records SET trust_state = 'evicted', evict_reason = ?, evict_at = ?
             WHERE node_id = ?"
        )
        .bind(reason)
        .bind(now)
        .bind(node_id)
        .execute(&self.pool)
        .await?;

        // Write revocation record for gossip propagation
        sqlx::query(
            "INSERT INTO revocations (node_id, revoked_by, reason, signature, revoked_at)
             VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(node_id) DO UPDATE SET
                revoked_by = excluded.revoked_by, reason = excluded.reason,
                signature  = excluded.signature,  revoked_at = excluded.revoked_at"
        )
        .bind(node_id)
        .bind(actor)
        .bind(reason)
        .bind(revocation_sig)
        .bind(now)
        .execute(&self.pool)
        .await?;

        // Purge every cache entry that originated from this node
        let purged = cache.purge_by_node_id(node_id).await.unwrap_or(0);

        self.log_event(node_id, "evicted", actor, Some(reason)).await;
        warn!("node {node_id} evicted by {actor}: {reason} — purged {purged} cache entries");
        Ok(())
    }

    /// Returns all revocations for gossip propagation.
    pub async fn list_revocations(&self) -> Result<Vec<RevocationRecord>> {
        let rows = sqlx::query(
            "SELECT node_id, revoked_by, reason, signature, revoked_at FROM revocations ORDER BY revoked_at DESC"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| RevocationRecord {
            node_id:    r.get("node_id"),
            revoked_by: r.get("revoked_by"),
            reason:     r.get("reason"),
            signature:  r.get("signature"),
            revoked_at: r.get("revoked_at"),
        }).collect())
    }

    // ── Peer health ─────────────────────────────────────────────────────────

    /// Record the outcome of a health check probe to a peer node.
    /// Uses an exponential moving average (α=0.2) for latency smoothing.
    /// A peer becomes unreachable after `failure_threshold` consecutive failures.
    pub async fn record_health_check(
        &self,
        node_id:           &str,
        url:               &str,
        success:           bool,
        latency_ms:        Option<u64>,
        failure_threshold: u32,
    ) -> Result<()> {
        let now = Utc::now().timestamp();
        let lat = latency_ms.map(|l| l as i64);
        let success_i = if success { 1i32 } else { 0i32 };

        // initial_is_reachable is only used for brand-new INSERT rows.
        // The ON CONFLICT UPDATE branch uses success_i (?3) directly.
        let initial_is_reachable = if success { 1i32 } else if failure_threshold <= 1 { 0i32 } else { 1i32 };
        let initial_cons_fail    = if success { 0i32 } else { 1i32 };
        let initial_cons_ok      = if success { 1i32 } else { 0i32 };

        sqlx::query(
            "INSERT INTO peer_health
                (node_id, url, is_reachable, latency_ms, avg_latency_ms,
                 last_checked, last_success, consecutive_fail, consecutive_ok, check_count)
             VALUES (?1, ?2, ?10, ?4, ?4, ?5, ?6, ?7, ?8, 1)
             ON CONFLICT(node_id) DO UPDATE SET
                url             = ?2,
                latency_ms      = ?4,
                avg_latency_ms  = CASE
                    WHEN ?4 IS NOT NULL AND avg_latency_ms IS NOT NULL
                         THEN avg_latency_ms * 0.8 + CAST(?4 AS REAL) * 0.2
                    WHEN ?4 IS NOT NULL THEN CAST(?4 AS REAL)
                    ELSE avg_latency_ms
                END,
                last_checked    = ?5,
                last_success    = CASE WHEN ?3 = 1 THEN ?5 ELSE last_success END,
                consecutive_fail = CASE WHEN ?3 = 0 THEN consecutive_fail + 1 ELSE 0 END,
                consecutive_ok  = CASE WHEN ?3 = 1 THEN consecutive_ok  + 1 ELSE 0 END,
                check_count     = check_count + 1,
                is_reachable    = CASE
                    WHEN ?3 = 1 THEN 1
                    WHEN consecutive_fail + 1 >= ?9 THEN 0
                    ELSE is_reachable
                END"
        )
        .bind(node_id)                  // ?1
        .bind(url)                      // ?2
        .bind(success_i)                // ?3 — actual success flag for UPDATE CASE expressions
        .bind(lat)                      // ?4
        .bind(now)                      // ?5
        .bind(if success { Some(now) } else { None::<i64> })  // ?6 last_success for INSERT
        .bind(initial_cons_fail)        // ?7
        .bind(initial_cons_ok)          // ?8
        .bind(failure_threshold as i64) // ?9
        .bind(initial_is_reachable)     // ?10 — initial is_reachable for INSERT only
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Returns health records for all known peers, ordered by average latency
    /// (fastest first, unknowns last).
    pub async fn list_peer_health(&self) -> Result<Vec<PeerHealth>> {
        let rows = sqlx::query(
            "SELECT node_id, url, is_reachable, latency_ms, avg_latency_ms,
                    last_checked, last_success, consecutive_fail, check_count
             FROM peer_health
             ORDER BY CASE WHEN avg_latency_ms IS NULL THEN 1 ELSE 0 END,
                      avg_latency_ms ASC"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| PeerHealth {
            node_id:          r.get("node_id"),
            url:              r.get("url"),
            is_reachable:     r.get::<i32, _>("is_reachable") != 0,
            latency_ms:       r.get("latency_ms"),
            avg_latency_ms:   r.get("avg_latency_ms"),
            last_checked:     r.get("last_checked"),
            last_success:     r.get("last_success"),
            consecutive_fail: r.get::<i32, _>("consecutive_fail") as u32,
            check_count:      r.get("check_count"),
        }).collect())
    }

    /// Quick check used by the federation client before attempting a peer lookup.
    /// Returns `true` if the peer has no health record yet (give benefit of the doubt)
    /// or if its last recorded state was reachable.
    pub async fn is_peer_reachable(&self, node_id: &str) -> bool {
        let row = sqlx::query("SELECT is_reachable FROM peer_health WHERE node_id = ?")
            .bind(node_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        match row {
            Some(r) => r.get::<i32, _>("is_reachable") != 0,
            None    => true, // no data yet — assume reachable
        }
    }

    async fn log_event(&self, node_id: &str, event: &str, actor: &str, reason: Option<&str>) {
        let id  = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();
        let _ = sqlx::query(
            "INSERT INTO trust_events (id, node_id, event, actor, reason, created_at) VALUES (?,?,?,?,?,?)"
        )
        .bind(&id)
        .bind(node_id)
        .bind(event)
        .bind(actor)
        .bind(reason)
        .bind(now)
        .execute(&self.pool)
        .await;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationRecord {
    pub node_id:    String,
    pub revoked_by: String,
    pub reason:     String,
    pub signature:  String,
    pub revoked_at: i64,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn row_to_record(row: sqlx::sqlite::SqliteRow) -> NodeRecord {
    let state: String = row.get("trust_state");
    let trust = match state.as_str() {
        "trusted" => NodeTrustState::Trusted {
            signed_by: row.get::<Option<String>, _>("signed_by")
                .unwrap_or_else(|| "manual".into()),
        },
        "evicted" => NodeTrustState::Evicted {
            reason: row.get::<Option<String>, _>("evict_reason").unwrap_or_default(),
            at:     row.get::<Option<i64>, _>("evict_at").unwrap_or(0),
        },
        _ => NodeTrustState::Untrusted,
    };
    NodeRecord {
        node_id:        row.get("node_id"),
        public_key_hex: row.get("public_key_hex"),
        url:            row.get("url"),
        is_head:        row.get::<i32, _>("is_head") != 0,
        trust:          trust,
        first_seen:     row.get("first_seen"),
        last_seen:      row.get("last_seen"),
    }
}
