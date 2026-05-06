use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions},
    Row,
};
use std::str::FromStr;
use tokio::sync::RwLock;

use crate::config::BudgetConfig;

#[derive(Debug, Clone)]
pub struct ModelPricing {
    pub input_per_1k:  f64,
    pub output_per_1k: f64,
}

impl ModelPricing {
    pub fn estimate_cost(&self, tokens_in: u32, tokens_out: u32) -> f64 {
        (tokens_in as f64 / 1000.0) * self.input_per_1k
            + (tokens_out as f64 / 1000.0) * self.output_per_1k
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DailySummary {
    pub date:         String,
    pub total_usd:    f64,
    pub api_calls:    i64,
    pub tokens_in:    i64,
    pub tokens_out:   i64,
}

#[derive(Debug, Clone)]
pub enum BudgetStatus {
    Ok { spent_usd: f64, limit_usd: f64 },
    Warning { spent_usd: f64, limit_usd: f64, pct: f64 },
    Exceeded { spent_usd: f64, limit_usd: f64 },
}

impl BudgetStatus {
    pub fn is_exceeded(&self) -> bool {
        matches!(self, BudgetStatus::Exceeded { .. })
    }
}

pub struct BudgetLedger {
    pool:    SqlitePool,
    cfg:     BudgetConfig,
    pricing: RwLock<ModelPricing>,
}

impl BudgetLedger {
    pub async fn open(cfg: BudgetConfig) -> Result<Self> {
        let opts = SqliteConnectOptions::from_str(&format!("sqlite:{}", cfg.db_path))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(opts)
            .await?;

        let ledger = BudgetLedger {
            pricing: RwLock::new(ModelPricing {
                input_per_1k:  cfg.input_per_1k_usd,
                output_per_1k: cfg.output_per_1k_usd,
            }),
            pool,
            cfg,
        };
        ledger.migrate().await?;
        Ok(ledger)
    }

    async fn migrate(&self) -> Result<()> {
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS spend_events (
                id          TEXT    PRIMARY KEY,
                model       TEXT    NOT NULL,
                tokens_in   INTEGER NOT NULL,
                tokens_out  INTEGER NOT NULL,
                cost_usd    REAL    NOT NULL,
                day         TEXT    NOT NULL,
                created_at  INTEGER NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_spend_day ON spend_events(day)")
            .execute(&self.pool).await?;

        Ok(())
    }

    pub async fn check(&self) -> Result<BudgetStatus> {
        let day = today_str();
        let spent: f64 = sqlx::query(
            "SELECT COALESCE(SUM(cost_usd), 0.0) as s FROM spend_events WHERE day = ?"
        )
        .bind(&day)
        .fetch_one(&self.pool)
        .await?
        .get("s");

        let limit = self.cfg.daily_limit_usd;
        let pct   = (spent / limit) * 100.0;

        if spent >= limit {
            Ok(BudgetStatus::Exceeded { spent_usd: spent, limit_usd: limit })
        } else if pct >= self.cfg.warn_at_pct as f64 {
            Ok(BudgetStatus::Warning { spent_usd: spent, limit_usd: limit, pct })
        } else {
            Ok(BudgetStatus::Ok { spent_usd: spent, limit_usd: limit })
        }
    }

    pub async fn record(&self, model: &str, tokens_in: u32, tokens_out: u32) -> Result<f64> {
        let pricing = self.pricing.read().await;
        let cost = pricing.estimate_cost(tokens_in, tokens_out);
        drop(pricing);

        let id  = uuid::Uuid::new_v4().to_string();
        let day = today_str();
        let now = Utc::now().timestamp();

        sqlx::query(
            "INSERT INTO spend_events (id, model, tokens_in, tokens_out, cost_usd, day, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&id)
        .bind(model)
        .bind(tokens_in as i64)
        .bind(tokens_out as i64)
        .bind(cost)
        .bind(&day)
        .bind(now)
        .execute(&self.pool)
        .await?;

        Ok(cost)
    }

    pub async fn daily_summary(&self, days: u32) -> Result<Vec<DailySummary>> {
        let rows = sqlx::query(
            "SELECT day,
                    SUM(cost_usd)   as total_usd,
                    COUNT(*)        as api_calls,
                    SUM(tokens_in)  as tokens_in,
                    SUM(tokens_out) as tokens_out
             FROM spend_events
             GROUP BY day
             ORDER BY day DESC
             LIMIT ?"
        )
        .bind(days as i64)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| DailySummary {
                date:       r.get("day"),
                total_usd:  r.get("total_usd"),
                api_calls:  r.get("api_calls"),
                tokens_in:  r.get("tokens_in"),
                tokens_out: r.get("tokens_out"),
            })
            .collect())
    }

    pub async fn update_pricing(&self, input_per_1k: f64, output_per_1k: f64) {
        let mut p = self.pricing.write().await;
        p.input_per_1k  = input_per_1k;
        p.output_per_1k = output_per_1k;
    }

    pub async fn current_pricing(&self) -> ModelPricing {
        self.pricing.read().await.clone()
    }
}

fn today_str() -> String {
    Utc::now().format("%Y-%m-%d").to_string()
}
