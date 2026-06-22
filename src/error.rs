use axum::http::StatusCode;
use thiserror::Error;

/// Typed proxy errors that propagate through `anyhow::Error` and can be
/// downcast at the HTTP layer via `e.downcast_ref::<ProxyError>()`.
///
/// Every variant carries a human-readable context string and maps to a
/// specific HTTP status code.  Use `code()` in log messages so operators
/// can correlate log entries with HTTP response bodies.
///
/// | Code    | Variant             | HTTP | Condition                                      |
/// |---------|---------------------|------|------------------------------------------------|
/// | CC-E001 | BackendAtCapacity   |  503 | claude_code pool full, queue timed out         |
/// | CC-E002 | BackendTimeout      |  504 | claude CLI / Anthropic API exceeded timeout    |
/// | CC-E003 | NoApiAccess         |  401 | OAuth token has no api.anthropic.com access    |
/// | CC-E004 | RateLimited         |  429 | Anthropic rate_limit_error or overloaded_error |
/// | CC-E005 | CreditExhausted     |  402 | Proxy's Anthropic credit balance is zero       |
/// | CC-E006 | BackendUnavailable  |  503 | claude CLI not in PATH or failed to spawn      |
/// | CC-E007 | BudgetExceeded      |  429 | Local daily spend limit reached                |
#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("CC-E001 backend_at_capacity: {0}")]
    BackendAtCapacity(String),
    #[error("CC-E002 backend_timeout: {0}")]
    BackendTimeout(String),
    #[error("CC-E003 no_api_access: {0}")]
    NoApiAccess(String),
    #[error("CC-E004 rate_limited: {0}")]
    RateLimited(String),
    #[error("CC-E005 credit_exhausted: {0}")]
    CreditExhausted(String),
    #[error("CC-E006 backend_unavailable: {0}")]
    BackendUnavailable(String),
    #[error("CC-E007 budget_exceeded: {0}")]
    BudgetExceeded(String),
}

impl ProxyError {
    /// Stable error code for log correlation and HTTP response bodies.
    pub fn code(&self) -> &'static str {
        match self {
            Self::BackendAtCapacity(_)  => "CC-E001",
            Self::BackendTimeout(_)     => "CC-E002",
            Self::NoApiAccess(_)        => "CC-E003",
            Self::RateLimited(_)        => "CC-E004",
            Self::CreditExhausted(_)    => "CC-E005",
            Self::BackendUnavailable(_) => "CC-E006",
            Self::BudgetExceeded(_)     => "CC-E007",
        }
    }

    /// Anthropic-compatible error type string for the `error.type` field.
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::BackendAtCapacity(_)  => "backend_at_capacity",
            Self::BackendTimeout(_)     => "backend_timeout",
            Self::NoApiAccess(_)        => "no_api_access",
            Self::RateLimited(_)        => "rate_limited",
            Self::CreditExhausted(_)    => "credit_exhausted",
            Self::BackendUnavailable(_) => "backend_unavailable",
            Self::BudgetExceeded(_)     => "budget_exceeded",
        }
    }

    /// HTTP status code to return to the client.
    pub fn http_status(&self) -> StatusCode {
        match self {
            Self::BackendAtCapacity(_)  => StatusCode::SERVICE_UNAVAILABLE,  // 503
            Self::BackendTimeout(_)     => StatusCode::GATEWAY_TIMEOUT,       // 504
            Self::NoApiAccess(_)        => StatusCode::UNAUTHORIZED,          // 401
            Self::RateLimited(_)        => StatusCode::TOO_MANY_REQUESTS,     // 429
            Self::CreditExhausted(_)    => StatusCode::PAYMENT_REQUIRED,      // 402
            Self::BackendUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,   // 503
            Self::BudgetExceeded(_)     => StatusCode::TOO_MANY_REQUESTS,     // 429
        }
    }

    /// Suggested `Retry-After` delay in seconds for transient errors.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Self::BackendAtCapacity(_)  => Some(5),
            Self::BackendTimeout(_)     => Some(10),
            Self::RateLimited(_)        => Some(60),
            Self::BudgetExceeded(_)     => Some(300),
            _                           => None,
        }
    }
}
