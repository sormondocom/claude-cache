use axum::{extract::{Path, Query, State}, response::{Html, IntoResponse}, Json};
use serde::Deserialize;
use serde_json::json;

use crate::server::SharedState;

// ── Graph search params ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct GraphSearchParams {
    #[serde(default)]
    pub q:      String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default = "default_graph_limit")]
    pub limit:  i64,
}
fn default_graph_limit() -> i64 { 25 }

pub async fn handle_dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

pub async fn handle_overview(State(state): State<SharedState>) -> impl IntoResponse {
    let cache_stats  = state.cache.stats().await.ok();
    let budget_check = state.budget.check().await.ok();

    let (spent, limit, budget_status) = match budget_check {
        Some(crate::budget::BudgetStatus::Ok      { spent_usd, limit_usd }) => (spent_usd, limit_usd, "ok"),
        Some(crate::budget::BudgetStatus::Warning { spent_usd, limit_usd, .. }) => (spent_usd, limit_usd, "warning"),
        Some(crate::budget::BudgetStatus::Exceeded{ spent_usd, limit_usd }) => (spent_usd, limit_usd, "exceeded"),
        None => (0.0, 0.0, "unknown"),
    };

    let pricing = state.budget.current_pricing().await;

    use std::sync::atomic::Ordering;
    Json(json!({
        "node_id":           state.node_id,
        "is_cnc":            state.is_cnc,
        "credits_exhausted": state.credits_exhausted.load(Ordering::Relaxed),
        "manual_bypass":     state.manual_bypass.load(Ordering::Relaxed),
        "federation": {
            "enabled":    state.federation.is_enabled(),
            "peer_count": state.federation.peer_count().await,
        },
        "cache": cache_stats.map(|s| json!({
            "total_entries":  s.total_entries,
            "total_hits":     s.total_hits,
            "shared_entries": s.shared_entries,
        })),
        "budget": {
            "status":         budget_status,
            "spent_usd":      spent,
            "limit_usd":      limit,
            "pct":            if limit > 0.0 { spent / limit * 100.0 } else { 0.0 },
            "input_per_1k":   pricing.input_per_1k,
            "output_per_1k":  pricing.output_per_1k,
        }
    }))
}

pub async fn handle_cache_entries(State(state): State<SharedState>) -> impl IntoResponse {
    // Return the latest 50 shared hashes for the dashboard
    match state.cache.list_shared_hashes(50, 0).await {
        Ok(hashes) => Json(json!({ "hashes": hashes, "count": hashes.len() })).into_response(),
        Err(e)     => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_trust_nodes(State(state): State<SharedState>) -> impl IntoResponse {
    match state.trust.list_all().await {
        Ok(nodes) => Json(json!({ "nodes": nodes })).into_response(),
        Err(e)    => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_peer_health(State(state): State<SharedState>) -> impl IntoResponse {
    match state.trust.list_peer_health().await {
        Ok(records) => Json(json!({ "peers": records })).into_response(),
        Err(e)      => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_routing_log(State(state): State<SharedState>) -> impl IntoResponse {
    let (recent, stats) = tokio::join!(
        state.cache.routing_log_recent(50),
        state.cache.routing_log_stats(86400),
    );
    match (recent, stats) {
        (Ok(entries), Ok(summary)) =>
            Json(json!({ "summary": summary, "recent": entries })).into_response(),
        (Err(e), _) | (_, Err(e)) =>
            Json(json!({ "error": e.to_string() })).into_response(),
    }
}

// ── Cache search ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CacheSearchParams {
    #[serde(default)]
    pub q:      Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub limit:  Option<i64>,
}

pub async fn handle_cache_search(
    State(state): State<SharedState>,
    Query(params): Query<CacheSearchParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(50).max(1).min(200);
    match state.cache.search_entries(params.q.as_deref(), params.domain.as_deref(), limit).await {
        Ok(entries) => Json(json!({ "entries": entries, "count": entries.len() })).into_response(),
        Err(e)      => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

// ── Learning management endpoints ──────────────────────────────────────────

pub async fn handle_learning_knowledge(State(state): State<SharedState>) -> impl IntoResponse {
    match state.cache.list_knowledge_docs().await {
        Ok(docs) => Json(json!({ "docs": docs, "count": docs.len() })).into_response(),
        Err(e)   => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_learning_thresholds(State(state): State<SharedState>) -> impl IntoResponse {
    match state.cache.load_threshold_overrides().await {
        Ok(map) => {
            let entries: Vec<_> = map.into_iter()
                .map(|((domain, intent), t)| json!({ "domain": domain, "intent": intent, "novelty_threshold": t }))
                .collect();
            Json(json!({ "thresholds": entries, "count": entries.len() })).into_response()
        }
        Err(e) => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_learning_feedback(State(state): State<SharedState>) -> impl IntoResponse {
    match state.cache.list_recent_feedback(100).await {
        Ok(rows) => Json(json!({ "feedback": rows, "count": rows.len() })).into_response(),
        Err(e)   => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_learning_contrasts(State(state): State<SharedState>) -> impl IntoResponse {
    match state.cache.list_recent_contrasts(50).await {
        Ok(pairs) => Json(json!({ "contrasts": pairs, "count": pairs.len() })).into_response(),
        Err(e)    => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_learning_distill(
    State(state): State<SharedState>,
    Path(domain): Path<String>,
) -> impl IntoResponse {
    match state.distiller.distill_domain(&domain).await {
        Ok(doc) => {
            Json(json!({
                "domain":  domain,
                "chars":   doc.len(),
                "content": doc,
            })).into_response()
        }
        Err(e) => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_learning_brain(State(state): State<SharedState>) -> impl IntoResponse {
    match state.cache.brain_snapshot(86400).await {
        Ok(snap) => Json(snap).into_response(),
        Err(e)   => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_learning_calibration(State(state): State<SharedState>) -> impl IntoResponse {
    match state.cache.calibration_summary(604_800).await {
        Ok(data) => Json(data).into_response(),
        Err(e)   => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_learning_draft_verify(State(state): State<SharedState>) -> impl IntoResponse {
    match state.cache.draft_verify_stats(86_400).await {
        Ok(data) => Json(data).into_response(),
        Err(e)   => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_learning_forgetting(State(state): State<SharedState>) -> impl IntoResponse {
    let max_mult = state.cfg.load().cache.forgetting_max_multiplier;
    match state.cache.forgetting_stats(max_mult).await {
        Ok(data) => Json(data).into_response(),
        Err(e)   => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

// ── Graph handlers ────────────────────────────────────────────────────────

pub async fn handle_graph_page() -> Html<&'static str> {
    Html(GRAPH_HTML)
}

pub async fn handle_graph_data(State(state): State<SharedState>) -> impl IntoResponse {
    const TTL: std::time::Duration = std::time::Duration::from_secs(30);
    // Release the lock before the DB query so concurrent requests don't block
    // each other for the full duration of the graph scan.
    {
        let cache = state.graph_cache.lock().await;
        if let Some((ref data, ts)) = *cache {
            if ts.elapsed() < TTL {
                return Json(data.clone()).into_response();
            }
        }
    }
    match state.cache.graph_data(86400).await {
        Ok(data) => {
            *state.graph_cache.lock().await = Some((data.clone(), std::time::Instant::now()));
            Json(data).into_response()
        }
        Err(e) => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_graph_search(
    State(state): State<SharedState>,
    Query(p): Query<GraphSearchParams>,
) -> impl IntoResponse {
    let domain = p.domain.as_deref();
    match state.cache.search_entries_for_graph(&p.q, domain, p.limit).await {
        Ok(results) => Json(json!({ "results": results, "count": results.len() })).into_response(),
        Err(e)      => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

pub async fn handle_graph_trace(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let novelty_threshold = state.cfg.load().routing.novelty_threshold;
    match state.cache.entry_trace(&id, novelty_threshold).await {
        Ok(trace) => Json(trace).into_response(),
        Err(e)    => Json(json!({ "error": e.to_string() })).into_response(),
    }
}

// ── Dashboard HTML ─────────────────────────────────────────────────────────

static DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>claude-cache</title>
<style>
  :root { --bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--green:#3fb950;--yellow:#d29922;--red:#f85149;--blue:#58a6ff; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:24px; }
  h1 { font-size:1.4rem; font-weight:600; margin-bottom:24px; color:var(--blue); }
  h2 { font-size:1rem; font-weight:600; margin-bottom:12px; color:var(--muted); }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin-bottom:24px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }
  .card-label { font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px; }
  .card-value { font-size:1.6rem; font-weight:700; }
  .green { color:var(--green); } .yellow { color:var(--yellow); } .red { color:var(--red); } .blue { color:var(--blue); }
  .budget-bar-wrap { background:var(--border); border-radius:4px; height:8px; margin-top:10px; overflow:hidden; }
  .budget-bar { height:100%; border-radius:4px; transition:width .3s; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  th { text-align:left; padding:8px 12px; color:var(--muted); font-weight:500; border-bottom:1px solid var(--border); }
  td { padding:8px 12px; border-bottom:1px solid var(--border); font-family:monospace; word-break:break-all; }
  .node-id { font-size:.75rem; color:var(--muted); margin-bottom:16px; }
  .status-ok { color:var(--green); } .status-warning { color:var(--yellow); } .status-exceeded { color:var(--red); }
  .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:.75rem; font-weight:600; }
  .badge-trusted  { background:#1a3a1a; color:var(--green); }
  .badge-untrusted{ background:#2a2a1a; color:var(--yellow); }
  .badge-evicted  { background:#3a1a1a; color:var(--red); }
  .badge-head     { background:#1a2a3a; color:var(--blue); margin-left:4px; }
  .btn { padding:4px 10px; border-radius:4px; font-size:.75rem; cursor:pointer; border:1px solid var(--border); background:var(--bg); color:var(--text); }
  .btn:hover { border-color:var(--blue); color:var(--blue); }
  .btn-danger:hover { border-color:var(--red); color:var(--red); }
  input[type=text] { background:var(--bg); border:1px solid var(--border); border-radius:4px; color:var(--text); padding:4px 8px; font-size:.85rem; }
  input[type=text]:focus { outline:none; border-color:var(--blue); }
  select { background:var(--bg); border:1px solid var(--border); border-radius:4px; color:var(--text); padding:4px 8px; font-size:.85rem; }
  .pin-badge { display:inline-block; padding:1px 6px; border-radius:8px; font-size:.7rem; background:#1a2a3a; color:var(--blue); }
  #toast { position:fixed; bottom:24px; right:24px; padding:10px 18px; border-radius:6px; font-size:.85rem; display:none; }
  #toast.ok  { background:var(--green); color:#000; }
  #toast.err { background:var(--red);   color:#fff; }
</style>
</head>
<body>
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px">
  <h1 style="margin:0">claude-cache</h1>
  <div style="display:flex;gap:8px">
    <a href="/graph" style="color:var(--muted);font-size:.8rem;text-decoration:none;border:1px solid var(--border);padding:4px 12px;border-radius:4px" onmouseover="this.style.borderColor='var(--blue)';this.style.color='var(--blue)'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--muted)'">Brain Graph</a>
    <a href="/chat" style="color:var(--muted);font-size:.8rem;text-decoration:none;border:1px solid var(--border);padding:4px 12px;border-radius:4px" onmouseover="this.style.borderColor='var(--green)';this.style.color='var(--green)'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--muted)'">&#x1F47B; Chat</a>
  </div>
</div>
<div class="node-id" id="node-id">node: loading...</div>
<div class="grid" id="cards"></div>
<div class="card">
  <div class="card-label">Budget</div>
  <div id="budget-text" style="font-size:.9rem">loading...</div>
  <div class="budget-bar-wrap"><div class="budget-bar" id="budget-bar" style="width:0%"></div></div>
</div>
<div class="card">
  <h2>Trust &amp; Nodes</h2>
  <table id="trust-table">
    <thead><tr>
      <th>Node ID</th><th>State</th><th>Health</th><th>Latency</th><th>URL</th><th>Actions</th>
    </tr></thead>
    <tbody id="trust-body"><tr><td colspan="6" style="color:var(--muted)">loading...</td></tr></tbody>
  </table>
</div>
<div class="card">
  <h2>Request Activity (24 h)</h2>
  <div id="routing-breakdown" style="margin-bottom:16px;font-size:.85rem;color:var(--muted)">loading...</div>
  <div id="routing-recent"></div>
</div>
<div class="card">
  <h2>Cache Entries</h2>
  <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
    <input type="text" id="search-q" placeholder="Search prompt text..." style="flex:1;min-width:180px">
    <select id="search-domain">
      <option value="">All domains</option>
      <option value="rust">Rust</option>
      <option value="python">Python</option>
      <option value="typescript">TypeScript</option>
      <option value="javascript">JavaScript</option>
      <option value="sql">SQL</option>
      <option value="shell">Shell</option>
      <option value="general">General</option>
    </select>
    <button class="btn" onclick="searchCache()">Search</button>
  </div>
  <table id="cache-table">
    <thead><tr>
      <th>Prompt</th><th>Domain</th><th>Intent</th><th>Hits</th><th>Model</th><th>Pinned</th><th>Actions</th>
    </tr></thead>
    <tbody id="cache-body"><tr><td colspan="7" style="color:var(--muted)">loading...</td></tr></tbody>
  </table>
</div>
<div class="card" id="endpoints-card">
  <h2>API Endpoints</h2>
  <div id="endpoints-list" style="font-size:.82rem;color:var(--muted)">loading...</div>
</div>
<div id="toast"></div>
<script>
function toast(msg, ok) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = ok ? 'ok' : 'err';
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 3000);
}

async function trustAction(nodeId, action) {
  try {
    let url, body;
    if (action === 'promote') {
      url  = '/v1/trust/' + nodeId;
      body = JSON.stringify({ is_head: false });
    } else {
      url  = '/v1/evict/' + nodeId;
      body = JSON.stringify({ reason: 'evicted from dashboard' });
    }
    const r = await fetch(url, { method:'POST', headers:{'content-type':'application/json'}, body });
    const j = await r.json();
    if (r.ok && j.ok) { toast(action + ' ok', true); refreshTrust(); }
    else { toast(j.error || 'failed', false); }
  } catch(e) { toast(String(e), false); }
}

async function refreshTrust() {
  try {
    const [td, hd] = await Promise.all([
      fetch('/api/trust').then(r => r.json()),
      fetch('/api/peers/health').then(r => r.json()).catch(() => ({ peers: [] })),
    ]);
    const nodes  = td.nodes  || [];
    const health = {};
    for (const h of (hd.peers || [])) health[h.node_id] = h;

    const tbody = document.getElementById('trust-body');
    if (!nodes.length) {
      tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted)">no nodes known</td></tr>';
      return;
    }
    tbody.innerHTML = nodes.map(n => {
      const st  = n.trust?.state ?? 'untrusted';
      const cls = st === 'trusted' ? 'badge-trusted' : st === 'evicted' ? 'badge-evicted' : 'badge-untrusted';
      const headBadge = n.is_head ? '<span class="badge badge-head">HEAD</span>' : '';
      const short = (s) => s ? s.slice(0,16) + '...' : '';

      const h = health[n.node_id];
      let healthCell, latencyCell;
      if (!h) {
        healthCell  = '<span style="color:var(--muted);font-size:.75rem">—</span>';
        latencyCell = '<span style="color:var(--muted);font-size:.75rem">—</span>';
      } else if (h.is_reachable) {
        const lat = h.avg_latency_ms != null ? h.avg_latency_ms.toFixed(0) + ' ms' : '?';
        healthCell  = '<span style="color:var(--green);font-size:.75rem">●&nbsp;up</span>';
        latencyCell = `<span style="font-size:.75rem">${lat}</span>`;
      } else {
        const fails = h.consecutive_fail;
        healthCell  = `<span style="color:var(--red);font-size:.75rem">● down (${fails}x)</span>`;
        latencyCell = '<span style="color:var(--muted);font-size:.75rem">—</span>';
      }

      const actions = st !== 'evicted' ? `
        ${st !== 'trusted' ? `<button class="btn" onclick="trustAction('${n.node_id}','promote')">Promote</button>` : ''}
        <button class="btn btn-danger" onclick="trustAction('${n.node_id}','evict')">Evict</button>
      ` : '<span style="color:var(--red);font-size:.75rem">evicted</span>';
      return `<tr>
        <td>${short(n.node_id)}</td>
        <td><span class="badge ${cls}">${st}</span>${headBadge}</td>
        <td>${healthCell}</td>
        <td>${latencyCell}</td>
        <td style="font-size:.75rem">${n.url || ''}</td>
        <td>${actions}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

const DECISION_COLORS = {
  exact_cache:    'var(--green)',
  semantic_cache: '#2ea043',
  local:          'var(--blue)',
  api:            'var(--yellow)',
  federation:     '#a855f7',
};
const DECISION_BADGE = {
  exact_cache:    'badge-trusted',
  semantic_cache: 'badge-trusted',
  local:          'badge-head',
  api:            'badge-untrusted',
  federation:     'badge-head',
};

function timeAgo(ts) {
  const s = Math.floor(Date.now() / 1000) - ts;
  if (s < 60)   return s + 's ago';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  return Math.floor(s / 3600) + 'h ago';
}

async function refreshRouting() {
  try {
    const d = await fetch('/api/routing').then(r => r.json());
    if (d.error) return;
    const s     = d.summary || {};
    const total = s.total_requests || 0;
    const decs  = s.by_decision   || [];

    const missReasons = s.by_miss_reason || [];
    document.getElementById('routing-breakdown').innerHTML =
      `<div style="margin-bottom:10px"><span style="color:var(--text);font-size:1rem;font-weight:600">${total}</span> requests in 24 h</div>` +
      decs.map(dec => `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:.8rem">
          <div style="width:110px;color:var(--text)">${dec.decision}</div>
          <div style="flex:1;background:var(--border);border-radius:3px;height:6px;overflow:hidden">
            <div style="width:${dec.pct.toFixed(1)}%;background:${DECISION_COLORS[dec.decision]||'var(--muted)'};height:100%"></div>
          </div>
          <div style="width:36px;text-align:right">${dec.pct.toFixed(0)}%</div>
          <div style="width:72px;text-align:right;color:var(--muted)">${dec.avg_latency_ms.toFixed(0)} ms avg</div>
          <div style="width:64px;text-align:right;color:var(--green)">${dec.saved_usd > 0 ? '$' + dec.saved_usd.toFixed(4) : ''}</div>
        </div>`).join('') +
      (missReasons.length ? `
        <div style="margin-top:14px;margin-bottom:6px;font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em">API miss reasons (24 h)</div>` +
        missReasons.map(m => `
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:.78rem">
            <div style="width:200px;color:var(--muted)">${m.reason}</div>
            <div style="flex:1;background:var(--border);border-radius:3px;height:5px;overflow:hidden">
              <div style="width:${m.pct.toFixed(1)}%;background:var(--yellow);height:100%"></div>
            </div>
            <div style="width:36px;text-align:right;color:var(--muted)">${m.pct.toFixed(0)}%</div>
            <div style="width:36px;text-align:right;color:var(--muted)">${m.count}</div>
          </div>`).join('') : '');

    const rows = (d.recent || []).slice(0, 25);
    if (!rows.length) {
      document.getElementById('routing-recent').innerHTML =
        '<p style="color:var(--muted);font-size:.85rem">no requests yet</p>';
      return;
    }
    document.getElementById('routing-recent').innerHTML = `
      <table style="margin-top:8px">
        <thead><tr>
          <th>Time</th><th>Shape</th><th>Decision</th><th>Miss Reason</th><th>Latency</th><th>Tokens</th><th>Saved</th>
        </tr></thead>
        <tbody>
          ${rows.map(r => `<tr>
            <td style="color:var(--muted)">${timeAgo(r.created_at)}</td>
            <td style="font-size:.75rem">${r.shape_key || ''}</td>
            <td><span class="badge ${DECISION_BADGE[r.decision]||'badge-untrusted'}">${r.decision}</span></td>
            <td style="font-size:.72rem;color:var(--muted)">${r.miss_reason || ''}</td>
            <td>${r.latency_ms} ms</td>
            <td style="color:var(--muted)">${r.tokens_in != null ? r.tokens_in : ''}</td>
            <td style="color:var(--green)">${r.saved_usd ? '$' + r.saved_usd.toFixed(4) : ''}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch(e) { console.error(e); }
}

async function cacheAction(id, action) {
  try {
    let url, method, body;
    if (action === 'delete') {
      url = '/v1/cache/entries/' + id; method = 'DELETE'; body = null;
    } else if (action === 'pin') {
      url = '/v1/cache/entries/' + id + '/pin'; method = 'POST'; body = JSON.stringify({ pinned: true });
    } else if (action === 'unpin') {
      url = '/v1/cache/entries/' + id + '/pin'; method = 'POST'; body = JSON.stringify({ pinned: false });
    }
    const r = await fetch(url, { method, headers: body ? {'content-type':'application/json'} : {}, body });
    const j = await r.json();
    if (r.ok && j.ok) { toast(action + ' ok', true); searchCache(); }
    else { toast(j.error || 'failed', false); }
  } catch(e) { toast(String(e), false); }
}

async function searchCache() {
  const q      = document.getElementById('search-q').value.trim();
  const domain = document.getElementById('search-domain').value;
  let url = '/api/cache/search?limit=50';
  if (q)      url += '&q=' + encodeURIComponent(q);
  if (domain) url += '&domain=' + encodeURIComponent(domain);
  try {
    const d = await fetch(url).then(r => r.json());
    const entries = d.entries || [];
    const tbody = document.getElementById('cache-body');
    if (!entries.length) {
      tbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted)">no entries found</td></tr>';
      return;
    }
    tbody.innerHTML = entries.map(e => {
      const snippet = (e.prompt_preview || '').slice(0, 80) + (e.prompt_preview?.length > 80 ? '…' : '');
      const pinBtn  = e.pinned
        ? `<button class="btn" onclick="cacheAction('${e.id}','unpin')">Unpin</button>`
        : `<button class="btn" onclick="cacheAction('${e.id}','pin')">Pin</button>`;
      return `<tr>
        <td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${(e.prompt_preview||'').replace(/"/g,'&quot;')}">${snippet}</td>
        <td style="font-size:.75rem">${e.domain || ''}</td>
        <td style="font-size:.75rem">${e.intent || ''}</td>
        <td style="text-align:right">${e.hit_count ?? 0}</td>
        <td style="font-size:.75rem">${e.model_used || ''}</td>
        <td>${e.pinned ? '<span class="pin-badge">pinned</span>' : ''}</td>
        <td style="white-space:nowrap">
          ${pinBtn}
          <button class="btn btn-danger" onclick="cacheAction('${e.id}','delete')">Delete</button>
        </td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

document.getElementById('search-q').addEventListener('keydown', e => { if (e.key === 'Enter') searchCache(); });

function renderEndpoints(isCnc, fedEnabled) {
  const base = window.location.origin;
  const groups = [
    {
      label: 'Proxy',
      rows: [
        ['POST', '/v1/messages', 'route prompt to cache / local / API'],
      ]
    },
    {
      label: 'Prompt Annotations',
      rows: [
        ['', '![direct]', 'bypass cache + local model, go straight to Anthropic API'],
        ['', '![good]',   'mark the previous response as satisfactory (quality signal)'],
        ['', '![bad]',    'mark the previous response as unsatisfactory (quality signal)'],
      ]
    },
    {
      label: 'Health',
      rows: [
        ['GET', '/health', 'liveness check — returns node_id'],
      ]
    },
    ...(fedEnabled ? [{
      label: 'Federation',
      rows: [
        ['POST', '/v1/federation/announce',    'peer announce / bootstrap'],
        ['GET',  '/v1/federation/peers',       'list known peers'],
        ['GET',  '/v1/federation/lookup/:hash','semantic hash lookup'],
        ['POST', '/v1/federation/semantic',    'semantic search across peers'],
        ['GET',  '/v1/federation/revocations', 'pull revocation list'],
        ['POST', '/v1/federation/revocations', 'push revocations to peer'],
      ]
    }] : []),
    {
      label: 'Portal (protected)',
      rows: [
        ['GET',  '/',                 'dashboard'],
        ['GET',  '/stats',            'raw stats JSON'],
        ['GET',  '/api/overview',     'node / budget / federation summary'],
        ['GET',  '/api/cache',        'shared cache hashes'],
        ['GET',  '/api/cache/search', 'search cache entries'],
        ['GET',  '/api/spend',        'spend history'],
        ['POST', '/api/pricing',      'update token pricing'],
        ['GET',  '/api/trust',        'list trusted nodes'],
        ['GET',  '/api/peers/health', 'peer health checks'],
        ['GET',  '/api/routing',       'routing log + stats'],
        ['POST', '/api/bypass/enable', 'enable manual bypass mode'],
        ['POST', '/api/bypass/disable','disable manual bypass mode'],
      ]
    },
    {
      label: 'Learning (protected)',
      rows: [
        ['GET',  '/api/learning/knowledge',       'distilled domain knowledge docs'],
        ['GET',  '/api/learning/thresholds',      'adaptive routing threshold overrides'],
        ['GET',  '/api/learning/feedback',        'recent ![good]/![bad] quality signals'],
        ['GET',  '/api/learning/contrasts',       'escalation contrast pairs (wrong vs correct)'],
        ['GET',  '/api/learning/brain',           'aggregate brain growth snapshot across all domains'],
        ['POST', '/api/learning/distill/:domain', 'manually trigger distillation for a domain'],
      ]
    },
    {
      label: 'Cache Management (protected)',
      rows: [
        ['GET',    '/v1/cache/export',            'download all cache entries as JSON'],
        ['POST',   '/v1/cache/seed',              'import / pre-warm cache entries'],
        ['POST',   '/v1/cache/entries/:id/pin',   'pin or unpin an entry'],
        ['DELETE', '/v1/cache/entries/:id',       'delete a cache entry'],
      ]
    },
    ...(isCnc ? [{
      label: 'Trust / Eviction (CNC only)',
      rows: [
        ['GET',  '/v1/trust',            'list trusted nodes'],
        ['POST', '/v1/trust/:node_id',   'promote peer to trusted'],
        ['POST', '/v1/evict/:node_id',   'evict peer and purge its cache'],
      ]
    }] : []),
  ];

  const METHOD_COLOR = { GET:'var(--blue)', POST:'var(--green)', DELETE:'var(--red)' };

  document.getElementById('endpoints-list').innerHTML = groups.map(g => `
    <div style="margin-bottom:14px">
      <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:6px">${g.label}</div>
      ${g.rows.map(([m, p, desc]) => `
        <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px;font-family:monospace">
          <span style="color:${METHOD_COLOR[m]||'var(--muted)'};min-width:52px;font-size:.75rem;font-weight:700">${m}</span>
          <a href="${base}${p}" target="_blank" style="color:var(--text);text-decoration:none;font-size:.8rem" onmouseover="this.style.color='var(--blue)'" onmouseout="this.style.color='var(--text)'">${p}</a>
          <span style="color:var(--muted);font-size:.75rem">${desc}</span>
        </div>`).join('')}
    </div>`).join('');
}

async function refresh() {
  try {
    const ov = await fetch('/api/overview').then(r => r.json());
    document.getElementById('node-id').textContent = 'node: ' + ov.node_id;
    renderEndpoints(ov.is_cnc, ov.federation?.enabled);
    const c  = ov.cache || {};
    const b  = ov.budget || {};
    const f  = ov.federation || {};
    document.getElementById('cards').innerHTML = `
      <div class="card"><div class="card-label">Cache Entries</div><div class="card-value blue">${c.total_entries ?? 0}</div></div>
      <div class="card"><div class="card-label">Cache Hits</div><div class="card-value green">${c.total_hits ?? 0}</div></div>
      <div class="card"><div class="card-label">Shared Entries</div><div class="card-value">${c.shared_entries ?? 0}</div></div>
      <div class="card"><div class="card-label">Federation Peers</div><div class="card-value ${f.enabled ? 'blue' : ''}">${f.peer_count ?? 0}</div></div>
    `;
    const pct  = b.pct ?? 0;
    const col  = pct >= 100 ? 'var(--red)' : pct >= 80 ? 'var(--yellow)' : 'var(--green)';
    document.getElementById('budget-text').innerHTML =
      `<span class="status-${b.status}">$${(b.spent_usd??0).toFixed(4)} / $${(b.limit_usd??0).toFixed(2)} (${pct.toFixed(1)}%)</span>`;
    const bar = document.getElementById('budget-bar');
    bar.style.width = Math.min(pct, 100) + '%';
    bar.style.background = col;
  } catch(e) { console.error(e); }

  refreshTrust();
  refreshRouting();
  searchCache();
}
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"#;

// ── Graph HTML ────────────────────────────────────────────────────────────────

static GRAPH_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Brain Graph — claude-cache</title>
<script src="https://unpkg.com/d3@7.9.0/dist/d3.min.js"></script>
<style>
:root {
  --bg:#0d1117; --surface:#161b22; --surface2:#1c2333; --border:#30363d;
  --text:#e6edf3; --muted:#8b949e; --blue:#58a6ff; --cyan:#39c5cf;
  --green:#3fb950; --yellow:#d29922; --red:#f85149; --orange:#f0883e;
  --purple:#bc8cff;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:13px}
#app{display:flex;flex-direction:column;height:100vh}

/* ── Top bar ── */
#topbar{height:44px;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 16px;gap:16px;flex-shrink:0;z-index:10}
#topbar a{color:var(--muted);text-decoration:none;font-size:12px}
#topbar a:hover{color:var(--blue)}
#topbar h1{font-size:14px;font-weight:600;color:var(--cyan);white-space:nowrap}
#hdr-stats{display:flex;gap:16px;margin-left:auto;color:var(--muted);font-size:12px}
#hdr-stats span{color:var(--text)}

/* ── Main layout ── */
#main{display:flex;flex:1;overflow:hidden}

/* ── Sidebar ── */
#sidebar{width:280px;border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
#search-area{padding:10px;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:6px}
#search-input{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:6px 10px;font-size:13px;outline:none}
#search-input:focus{border-color:var(--blue)}
#domain-filter{background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--muted);padding:4px 8px;font-size:12px;outline:none;cursor:pointer}
#results-wrap{flex:1;overflow-y:auto;padding:6px}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin-bottom:6px;cursor:pointer;transition:border-color .15s}
.result-card:hover,.result-card.active{border-color:var(--blue)}
.result-card.active{background:var(--surface2)}
.rc-badges{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px}
.badge{display:inline-block;padding:1px 6px;border-radius:10px;font-size:11px;font-weight:600}
.b-domain{background:#1a2a3a;color:var(--blue)}
.b-intent{background:#1a3a2a;color:var(--green)}
.b-local{background:#2a1a3a;color:var(--purple)}
.b-api{background:#3a1a1a;color:var(--red)}
.b-cache{background:#1a3a1a;color:var(--green)}
.b-fed{background:#2a2a1a;color:var(--yellow)}
.rc-preview{color:var(--muted);font-size:11px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.rc-meta{display:flex;gap:8px;margin-top:4px;color:var(--muted);font-size:11px}
.rc-meta .hits{color:var(--yellow)}
.rc-meta .conf{color:var(--purple)}
.no-results{color:var(--muted);padding:16px;text-align:center;font-size:12px}

/* ── Graph canvas ── */
#graph-wrap{flex:1;position:relative;overflow:hidden}
#graph-svg{width:100%;height:100%}
.link{stroke:var(--border);stroke-opacity:.7;fill:none}
.node-domain circle.main{cursor:pointer}
.node-intent circle.main{cursor:pointer;opacity:.9}
.node-label{fill:var(--text);font-size:11px;pointer-events:none;text-anchor:middle;dominant-baseline:central}
.node-label.domain-label{font-weight:600;font-size:12px}
.glow-ring{fill:none;stroke:var(--cyan);stroke-width:2.5;opacity:.55;animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.35}50%{opacity:.75}}
.contrast-dot{fill:var(--orange)}

/* ── Legend ── */
#legend{position:absolute;bottom:16px;left:16px;background:rgba(13,17,23,.88);border:1px solid var(--border);border-radius:8px;padding:10px 14px;display:flex;flex-direction:column;gap:6px;pointer-events:none}
.leg-row{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted)}
.leg-circle{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.leg-ring{width:14px;height:14px;border-radius:50%;border:2px solid var(--cyan);flex-shrink:0}
.leg-dot{width:7px;height:7px;border-radius:50%;background:var(--orange);flex-shrink:0}

/* ── Tooltip ── */
#tooltip{position:fixed;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-size:12px;pointer-events:none;z-index:100;max-width:240px;box-shadow:0 8px 24px rgba(0,0,0,.5);display:none}
#tooltip .tt-title{font-weight:600;color:var(--blue);margin-bottom:6px}
#tooltip .tt-row{display:flex;justify-content:space-between;gap:16px;margin-bottom:3px;color:var(--muted)}
#tooltip .tt-row span:last-child{color:var(--text);font-variant-numeric:tabular-nums}

/* ── Trace panel ── */
#trace-panel{width:0;border-left:1px solid transparent;display:flex;flex-direction:column;overflow:hidden;transition:width .25s ease,border-color .25s;flex-shrink:0}
#trace-panel.open{width:360px;border-color:var(--border)}
#trace-header{padding:10px 14px;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
#trace-header .th-title{font-weight:600;font-size:13px;color:var(--cyan)}
#trace-close{background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:0 4px}
#trace-close:hover{color:var(--text)}
#trace-body{flex:1;overflow-y:auto;padding:14px}
.trace-note{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:7px 10px;font-size:11px;color:var(--muted);margin-bottom:14px;line-height:1.5}
.t-step{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:4px}
.t-step.highlight{border-color:var(--blue)}
.t-step.outcome-local{border-color:var(--purple)}
.t-step.outcome-api{border-color:var(--orange)}
.t-step.outcome-cache{border-color:var(--green)}
.t-step-head{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.t-icon{font-size:16px}
.t-title{font-weight:600;font-size:12px}
.t-connector{display:flex;justify-content:center;height:16px;position:relative}
.t-connector::before{content:'';position:absolute;left:50%;top:0;bottom:0;border-left:2px solid var(--border);transform:translateX(-50%)}
.t-connector.thick::before{border-left-width:3px;border-color:var(--blue)}
.t-connector .arrow{position:absolute;bottom:0;left:50%;transform:translateX(-50%);color:var(--border);font-size:10px}
.score-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:11px}
.score-bar-row .sb-label{width:80px;color:var(--muted);flex-shrink:0}
.score-bar-wrap{flex:1;background:var(--surface2);border-radius:3px;height:7px;overflow:hidden}
.score-bar-fill{height:100%;border-radius:3px;transition:width .4s}
.sb-pass .score-bar-fill{background:var(--green)}
.sb-fail .score-bar-fill{background:var(--red)}
.sb-warn .score-bar-fill{background:var(--yellow)}
.score-bar-row .sb-val{width:70px;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.sb-pass .sb-val::after{content:' ✓';color:var(--green)}
.sb-fail .sb-val::after{content:' ✗';color:var(--red)}
.t-kv{display:flex;gap:8px;margin-bottom:4px;font-size:11px;flex-wrap:wrap}
.t-kv .tk{color:var(--muted)}
.t-kv .tv{color:var(--text);font-variant-numeric:tabular-nums}
.t-kv .tv.good{color:var(--green)}
.t-kv .tv.warn{color:var(--yellow)}
.t-kv .tv.bad{color:var(--red)}
.t-kv .tv.pur{color:var(--purple)}
.layer-badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:11px;font-weight:600;margin-right:4px;margin-bottom:4px}
.lb-l1{background:#1a2a3a;color:var(--blue)}
.lb-l2{background:#0a2a2a;color:var(--cyan)}
.lb-l3{background:#2a1a3a;color:var(--purple)}
.lb-l5{background:#3a1a0a;color:var(--orange)}
.conf-bar{display:flex;align-items:center;gap:8px;margin-top:6px}
.conf-track{flex:1;background:var(--surface2);border-radius:4px;height:10px;overflow:hidden}
.conf-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--red) 0%,var(--yellow) 50%,var(--green) 80%)}
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px}
.stat-box{background:var(--surface2);border-radius:6px;padding:6px 8px;text-align:center}
.stat-val{font-size:16px;font-weight:700;color:var(--text)}
.stat-lbl{font-size:10px;color:var(--muted);margin-top:2px}
</style>
</head>
<body>
<div id="app">
  <div id="topbar">
    <a href="/">← Dashboard</a>
    <h1>Brain Knowledge Graph</h1>
    <div id="hdr-stats">
      <div>Entries <span id="hs-entries">—</span></div>
      <div>Domains <span id="hs-domains">—</span></div>
      <div>24 h requests <span id="hs-req">—</span></div>
    </div>
  </div>
  <div id="main">
    <div id="sidebar">
      <div id="search-area">
        <input id="search-input" type="text" placeholder="Search prompts, topics…" autocomplete="off">
        <select id="domain-filter"><option value="">All domains</option></select>
      </div>
      <div id="results-wrap"><p class="no-results">Start typing to search, or click a node</p></div>
    </div>
    <div id="graph-wrap">
      <svg id="graph-svg"></svg>
      <div id="legend">
        <div class="leg-row"><div class="leg-circle" style="background:#3fb950"></div>Low escalation (≤25%)</div>
        <div class="leg-row"><div class="leg-circle" style="background:#d29922"></div>Medium (25–70%)</div>
        <div class="leg-row"><div class="leg-circle" style="background:#f85149"></div>High (>70%)</div>
        <div class="leg-row"><div class="leg-ring"></div>L2 knowledge doc</div>
        <div class="leg-row"><div class="leg-dot"></div>Contrast pairs stored</div>
      </div>
    </div>
    <div id="trace-panel">
      <div id="trace-header">
        <span class="th-title">Decision Trace</span>
        <button id="trace-close">✕</button>
      </div>
      <div id="trace-body"></div>
    </div>
  </div>
</div>
<div id="tooltip"></div>

<script>
// ── Utilities ────────────────────────────────────────────────────────────────
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const pct = (v,d=0) => (v*100).toFixed(d)+'%';
const fmt2 = v => (v??0).toFixed(2);
const fmtAge = ts => {
  const d = Math.floor((Date.now()/1000 - ts)/86400);
  return d===0 ? 'today' : d===1 ? '1d ago' : d+'d ago';
};
const escalationColor = (() => {
  const scale = d3.scaleLinear()
    .domain([0, .25, .70, 1])
    .range(["#3fb950","#a3e635","#d29922","#f85149"])
    .clamp(true);
  return v => scale(v);
})();
const modelBadge = m => {
  if(m==='ollama')    return '<span class="badge b-local">local</span>';
  if(m==='anthropic') return '<span class="badge b-api">api</span>';
  if(m==='federated') return '<span class="badge b-fed">fed</span>';
  return '<span class="badge b-cache">cache</span>';
};

// ── Graph ────────────────────────────────────────────────────────────────────
let graphData = {nodes:[], links:[]};
let selectedDomain = null;
let sim;

const svg = d3.select('#graph-svg');
const container = document.getElementById('graph-wrap');

function getSize() {
  const r = container.getBoundingClientRect();
  return { w: r.width, h: r.height };
}

const gRoot = svg.append('g');
svg.call(d3.zoom().scaleExtent([.15, 4]).on('zoom', e => gRoot.attr('transform', e.transform)));

let linkSel, domainSel, intentSel;

function buildGraph(data) {
  graphData = data;
  const {w, h} = getSize();
  gRoot.selectAll('*').remove();

  const domainNodes = data.nodes.filter(n => n.type==='domain');
  const intentNodes = data.nodes.filter(n => n.type==='intent');

  // Populate domain filter dropdown
  const df = document.getElementById('domain-filter');
  const existing = new Set([...df.options].map(o=>o.value));
  domainNodes.forEach(d => {
    if(!existing.has(d.label)) {
      const o = document.createElement('option');
      o.value = d.label; o.textContent = d.label;
      df.appendChild(o);
    }
  });

  // Header stats
  const totalEntries = domainNodes.reduce((a,n)=>a+n.entries,0);
  const totalReq = domainNodes.reduce((a,n)=>a+n.requests_24h,0);
  document.getElementById('hs-entries').textContent = totalEntries;
  document.getElementById('hs-domains').textContent = domainNodes.length;
  document.getElementById('hs-req').textContent = totalReq;

  const nodeRadius = d => d.type==='domain'
    ? Math.max(20, 14 + Math.sqrt(d.entries) * 2.2)
    : Math.max(8,  5  + Math.sqrt(d.entries) * 1.2);

  // Force simulation
  sim = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(data.links).id(d=>d.id)
      .distance(d => 90 + Math.max(0, 60 - Math.log1p(d.value)*12))
      .strength(.45))
    .force('charge', d3.forceManyBody()
      .strength(d => d.type==='domain' ? -520 : -120)
      .distanceMax(340))
    .force('center', d3.forceCenter(w/2, h/2).strength(.04))
    .force('collide', d3.forceCollide().radius(d => nodeRadius(d)+10).strength(.75))
    .alphaDecay(.025);

  // Links
  linkSel = gRoot.append('g').selectAll('line').data(data.links).join('line')
    .attr('class','link')
    .attr('stroke-width', d => Math.max(1.2, Math.log1p(d.value) * 1.5))
    .attr('stroke-opacity', d => Math.min(.85, .3 + Math.log1p(d.value)*.08));

  // Domain nodes
  domainSel = gRoot.append('g').selectAll('g').data(domainNodes).join('g')
    .attr('class','node-domain')
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }))
    .on('click',  (_,d) => onDomainClick(d))
    .on('mouseover', (e,d) => showTooltip(e, buildDomainTooltip(d)))
    .on('mousemove',  e   => moveTooltip(e))
    .on('mouseout',   _   => hideTooltip());

  // L2 glow ring
  domainSel.filter(d => d.has_doc)
    .append('circle').attr('class','glow-ring')
    .attr('r', d => nodeRadius(d)+7);

  // Main circle
  domainSel.append('circle').attr('class','main')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => escalationColor(d.escalation_rate))
    .attr('stroke', '#0d1117').attr('stroke-width', 2);

  // Contrast dot badge
  domainSel.filter(d => d.contrast_pairs > 0)
    .append('circle').attr('class','contrast-dot')
    .attr('r', 5)
    .attr('cx', d => nodeRadius(d) - 4)
    .attr('cy', d => -nodeRadius(d) + 4);

  // Label
  domainSel.append('text').attr('class','node-label domain-label')
    .text(d => d.label);

  // Intent nodes
  intentSel = gRoot.append('g').selectAll('g').data(intentNodes).join('g')
    .attr('class','node-intent')
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }))
    .on('click',  (_,d) => onIntentClick(d))
    .on('mouseover', (e,d) => showTooltip(e, buildIntentTooltip(d)))
    .on('mousemove',  e   => moveTooltip(e))
    .on('mouseout',   _   => hideTooltip());

  intentSel.append('circle').attr('class','main')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => escalationColor(d.escalation_rate))
    .attr('stroke', '#0d1117').attr('stroke-width', 1.5);

  intentSel.append('text').attr('class','node-label')
    .text(d => d.label)
    .attr('dy', d => nodeRadius(d) + 12);

  sim.on('tick', () => {
    linkSel
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    domainSel.attr('transform', d => `translate(${d.x},${d.y})`);
    intentSel.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

// ── Tooltips ─────────────────────────────────────────────────────────────────
const ttEl = document.getElementById('tooltip');
function showTooltip(e, html) {
  ttEl.innerHTML = html;
  ttEl.style.display = 'block';
  moveTooltip(e);
}
function moveTooltip(e) {
  const x = e.clientX + 14, y = e.clientY - 10;
  ttEl.style.left = Math.min(x, window.innerWidth - 260) + 'px';
  ttEl.style.top  = Math.min(y, window.innerHeight - 200) + 'px';
}
function hideTooltip() { ttEl.style.display='none'; }

function row(k,v) { return `<div class="tt-row"><span>${k}</span><span>${v}</span></div>`; }
function buildDomainTooltip(d) {
  return `<div class="tt-title">${d.label}</div>
    ${row('Entries', d.entries)}
    ${row('24h escalation', pct(d.escalation_rate,0))}
    ${row('24h requests', d.requests_24h)}
    ${d.has_doc ? row('L2 doc', `v${d.doc_version}, ${d.doc_chars} chars`) : ''}
    ${d.contrast_pairs>0 ? row('Contrast pairs', d.contrast_pairs) : ''}
    ${(d.feedback_good+d.feedback_bad)>0 ? row('Feedback', `+${d.feedback_good} / -${d.feedback_bad}`) : ''}`;
}
function buildIntentTooltip(d) {
  const dir = d.adapted ? (d.threshold < d.base_threshold ? '↓ lowered' : '↑ raised') : 'base';
  return `<div class="tt-title">${d.domain}:<strong>${d.label}</strong></div>
    ${row('Entries', d.entries)}
    ${row('24h escalation', pct(d.escalation_rate,0))}
    ${row('Threshold', `${d.threshold.toFixed(2)} (${dir})`)}
    ${row('24h requests', d.requests_24h)}`;
}

// ── Node click → filter sidebar ──────────────────────────────────────────────
function onDomainClick(d) {
  selectedDomain = (selectedDomain === d.label) ? null : d.label;
  document.getElementById('domain-filter').value = selectedDomain || '';
  doSearch();
  // Dim non-selected domains
  domainSel.selectAll('circle.main')
    .attr('opacity', n => (!selectedDomain || n.label===selectedDomain) ? 1 : .25);
  intentSel.selectAll('circle.main')
    .attr('opacity', n => (!selectedDomain || n.domain===selectedDomain) ? 1 : .2);
  linkSel.attr('stroke-opacity', l => {
    if(!selectedDomain) return Math.min(.85, .3+Math.log1p(l.value)*.08);
    const srcDom = l.source.id && l.source.id.startsWith('domain:') ? l.source.label : l.source.domain;
    return srcDom===selectedDomain ? .85 : .1;
  });
}
function onIntentClick(d) {
  document.getElementById('domain-filter').value = d.domain;
  selectedDomain = d.domain;
  document.getElementById('search-input').value = d.label;
  doSearch();
}

// ── Search ───────────────────────────────────────────────────────────────────
let searchTimer = null;
document.getElementById('search-input').addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(doSearch, 280);
});
document.getElementById('domain-filter').addEventListener('change', e => {
  selectedDomain = e.target.value || null;
  doSearch();
});

async function doSearch() {
  const q = document.getElementById('search-input').value.trim();
  const domain = document.getElementById('domain-filter').value;
  const wrap = document.getElementById('results-wrap');
  if(!q && !domain) {
    wrap.innerHTML = '<p class="no-results">Start typing to search, or click a node</p>';
    return;
  }
  const params = new URLSearchParams({q, limit:25});
  if(domain) params.set('domain', domain);
  try {
    const r = await fetch('/api/graph/search?'+params);
    const data = await r.json();
    renderResults(data.results || []);
  } catch(e) {
    wrap.innerHTML = '<p class="no-results">Search error</p>';
  }
}

function renderResults(results) {
  const wrap = document.getElementById('results-wrap');
  if(!results.length) {
    wrap.innerHTML = '<p class="no-results">No matching entries</p>';
    return;
  }
  wrap.innerHTML = results.map(r => `
    <div class="result-card" data-id="${r.id}" onclick="loadTrace('${r.id}')">
      <div class="rc-badges">
        <span class="badge b-domain">${esc(r.domain)}</span>
        <span class="badge b-intent">${esc(r.intent)}</span>
        ${modelBadge(r.model_used)}
      </div>
      <div class="rc-preview">${esc(r.preview)}</div>
      <div class="rc-meta">
        <span class="hits">↺ ${r.hit_count}</span>
        ${r.confidence!=null ? `<span class="conf">conf ${(r.confidence*100).toFixed(0)}%</span>` : ''}
        <span>${fmtAge(r.created_at)}</span>
        <span>cplx ${(r.complexity*100).toFixed(0)}%</span>
      </div>
    </div>`).join('');
}

// ── Trace panel ──────────────────────────────────────────────────────────────
async function loadTrace(id) {
  document.querySelectorAll('.result-card').forEach(c => c.classList.toggle('active', c.dataset.id===id));
  const panel = document.getElementById('trace-panel');
  const body  = document.getElementById('trace-body');
  panel.classList.add('open');
  body.innerHTML = '<p class="no-results">Loading…</p>';
  try {
    const r = await fetch(`/api/graph/trace/${id}`);
    const data = await r.json();
    if(data.error) { body.innerHTML = `<p class="no-results">${esc(data.error)}</p>`; return; }
    body.innerHTML = renderTrace(data);
    // Highlight matching node in graph
    const domain = data.entry?.domain;
    if(domain) highlightDomain(domain);
  } catch(e) {
    body.innerHTML = '<p class="no-results">Error loading trace</p>';
  }
}

document.getElementById('trace-close').addEventListener('click', () => {
  document.getElementById('trace-panel').classList.remove('open');
  document.querySelectorAll('.result-card').forEach(c => c.classList.remove('active'));
});

function highlightDomain(domain) {
  domainSel.selectAll('circle.main')
    .attr('stroke', n => n.label===domain ? '#58a6ff' : '#0d1117')
    .attr('stroke-width', n => n.label===domain ? 3 : 2);
}

// ── Trace renderer ───────────────────────────────────────────────────────────
function scoreBarHtml(label, value, threshold, invert=false) {
  const pctVal = Math.min(100, (value*100).toFixed(1));
  const passed = value < threshold;
  const cls    = passed ? 'sb-pass' : 'sb-fail';
  return `<div class="score-bar-row ${cls}">
    <span class="sb-label">${label}</span>
    <div class="score-bar-wrap"><div class="score-bar-fill" style="width:${pctVal}%"></div></div>
    <span class="sb-val">${value.toFixed(2)} / ${threshold.toFixed(2)}</span>
  </div>`;
}

function confBarHtml(conf) {
  if(conf==null) return '';
  const p = (conf*100).toFixed(0);
  const col = conf>=.9 ? 'var(--green)' : conf>=.75 ? 'var(--yellow)' : 'var(--red)';
  return `<div class="conf-bar">
    <span style="font-size:11px;color:var(--muted);width:80px">Confidence</span>
    <div class="conf-track"><div class="conf-fill" style="width:${p}%;background:${col}"></div></div>
    <span style="font-size:11px;width:40px;text-align:right;color:${col}">${p}%</span>
  </div>`;
}

function renderTrace(t) {
  const e  = t.entry;
  const sc = t.scores;
  const th = sc.threshold;
  const m  = e.model_used;

  const outcomeClass = m==='ollama' ? 'outcome-local' : m==='anthropic' ? 'outcome-api' : 'outcome-cache';
  const outcomeIcon  = m==='ollama' ? '🧠' : m==='anthropic' ? '☁️' : '⚡';
  const outcomeLabel = m==='ollama' ? 'Served by Local Model' : m==='anthropic' ? 'Served by Anthropic API' : 'Served from Cache';

  // Estimate whether gate passed (if model=ollama it passed; if api, need miss_reason)
  const gatePassed = m==='ollama';
  const gateAllPass = sc.novelty_est < th && sc.complexity < .4 && sc.consequence < .3;

  // Layer 3 info
  const thrInfo = t.threshold;
  const adapted = thrInfo?.adapted;

  // Layer 2 info
  const hasDoc = !!t.knowledge_doc;

  // Layer 1 estimate: how many similar entries exist
  const poolSize = t.similar_count ?? 0;
  const estShots = Math.min(3, Math.floor(poolSize * .15));

  // Contrast pairs
  const hasPairs = t.contrast_pairs?.length > 0;

  // Routing stats
  const rs = t.routing_stats;

  return `
<div class="trace-note">
  Reconstructed trace — scores are estimates from entry metadata.
  Layer state reflects current system (not time of routing).
</div>

<!-- Step 1: Prompt -->
<div class="t-step">
  <div class="t-step-head"><span class="t-icon">📝</span><span class="t-title">Prompt</span></div>
  <div class="t-kv"><span class="tk">Preview</span></div>
  <div style="font-size:11px;color:var(--muted);line-height:1.5;margin-top:2px;font-style:italic">
    "${esc((e.preview||'').substring(0,120))}…"
  </div>
</div>
<div class="t-connector thick"><span class="arrow">▼</span></div>

<!-- Step 2: Classification -->
<div class="t-step">
  <div class="t-step-head"><span class="t-icon">🏷️</span><span class="t-title">Classification</span></div>
  <div class="rc-badges" style="margin-bottom:6px">
    <span class="badge b-domain">${esc(e.domain)}</span>
    <span class="badge b-intent">${esc(e.intent)}</span>
  </div>
  <div class="score-bar-row sb-warn">
    <span class="sb-label">Complexity</span>
    <div class="score-bar-wrap"><div class="score-bar-fill" style="width:${(sc.complexity*100).toFixed(0)}%;background:var(--yellow)"></div></div>
    <span class="sb-val" style="color:var(--muted)">${sc.complexity.toFixed(2)}</span>
  </div>
</div>
<div class="t-connector${gatePassed ? ' thick' : ''}"><span class="arrow">▼</span></div>

<!-- Step 3: Routing Gate -->
<div class="t-step${gatePassed ? ' highlight' : ''}">
  <div class="t-step-head"><span class="t-icon">⚖️</span><span class="t-title">Routing Gate</span>
    ${adapted ? `<span class="layer-badge lb-l3">L3 adapted</span>` : ''}
  </div>
  ${scoreBarHtml('Novelty', sc.novelty_est, th)}
  ${scoreBarHtml('Complexity', sc.complexity, .4)}
  ${scoreBarHtml('Consequence', sc.consequence, .3)}
  <div class="t-kv" style="margin-top:6px">
    <span class="tk">Threshold</span>
    <span class="tv">${th.toFixed(2)}${adapted ? ` (base ${thrInfo.base.toFixed(2)}, adapted)` : ' (config base)'}</span>
  </div>
  ${thrInfo?.escalation_rate!=null ? `<div class="t-kv">
    <span class="tk">Domain esc. rate</span>
    <span class="tv ${thrInfo.escalation_rate > .7 ? 'bad' : thrInfo.escalation_rate < .3 ? 'good' : 'warn'}">
      ${pct(thrInfo.escalation_rate,0)} (n=${thrInfo.sample_count})
    </span>
  </div>` : ''}
  <div class="t-kv" style="margin-top:4px">
    <span class="tk">Verdict</span>
    <span class="tv ${gatePassed ? 'good' : 'bad'}">${gatePassed ? '✓ Proceed to local model' : '✗ Escalate to API'}</span>
  </div>
</div>
<div class="t-connector${gatePassed ? ' thick' : ''}"><span class="arrow">▼</span></div>

${gatePassed ? `
<!-- Step 4: Learning context -->
<div class="t-step">
  <div class="t-step-head"><span class="t-icon">📚</span><span class="t-title">Learning Context Injected</span></div>
  <div style="margin-bottom:6px">
    ${hasDoc  ? `<span class="layer-badge lb-l2">L2 doc ${t.knowledge_doc.doc_chars} chars v${t.knowledge_doc.version}</span>` : '<span style="font-size:11px;color:var(--muted)">No L2 doc yet</span>'}
    ${estShots>0 ? `<span class="layer-badge lb-l1">L1 ~${estShots} shots</span>` : ''}
    ${hasPairs  ? `<span class="layer-badge lb-l5">L5 ${t.contrast_pairs.length} contrast pairs</span>` : ''}
  </div>
  ${hasDoc ? `<div class="t-kv">
    <span class="tk">Knowledge doc</span>
    <span class="tv good">v${t.knowledge_doc.version}, synthesised from ${t.knowledge_doc.entry_count} entries</span>
  </div>` : ''}
  ${poolSize > 0 ? `<div class="t-kv">
    <span class="tk">Pool for L1</span>
    <span class="tv">${poolSize} similar entries in domain</span>
  </div>` : ''}
</div>
<div class="t-connector thick"><span class="arrow">▼</span></div>

<!-- Step 5: Local model -->
<div class="t-step outcome-local">
  <div class="t-step-head"><span class="t-icon">🧠</span><span class="t-title">Local Model (Ollama)</span></div>
  ${confBarHtml(e.confidence)}
  <div class="t-kv" style="margin-top:6px">
    <span class="tk">Confidence floor</span><span class="tv">0.75</span>
  </div>
  <div class="t-kv">
    <span class="tk">Result</span>
    <span class="tv ${(e.confidence??0)>=.75 ? 'good' : 'bad'}">
      ${(e.confidence??0)>=.75 ? '✓ Above floor — served' : '✗ Below floor — escalated to API'}
    </span>
  </div>
</div>
<div class="t-connector thick"><span class="arrow">▼</span></div>
` : ''}

<!-- Final: decision -->
<div class="t-step ${outcomeClass}">
  <div class="t-step-head"><span class="t-icon">${outcomeIcon}</span><span class="t-title">${outcomeLabel}</span></div>
  <div class="t-kv"><span class="tk">Cache hits</span><span class="tv good">${e.hit_count} times served from cache</span></div>
  ${rs ? `
  <div class="stats-grid">
    <div class="stat-box"><div class="stat-val" style="color:var(--green)">${rs.cache_24h}</div><div class="stat-lbl">cache (24h)</div></div>
    <div class="stat-box"><div class="stat-val" style="color:var(--purple)">${rs.local_24h}</div><div class="stat-lbl">local (24h)</div></div>
    <div class="stat-box"><div class="stat-val" style="color:var(--red)">${rs.api_24h}</div><div class="stat-lbl">api (24h)</div></div>
    <div class="stat-box"><div class="stat-val" style="color:var(--muted)">${rs.avg_latency_ms ? Math.round(rs.avg_latency_ms)+'ms' : '—'}</div><div class="stat-lbl">avg latency</div></div>
  </div>` : ''}
</div>

${t.contrast_pairs?.length ? `
<div style="margin-top:14px">
  <div style="font-size:11px;font-weight:600;color:var(--orange);margin-bottom:6px">⚡ L5 Contrast Pairs — Failure Lessons for this Domain</div>
  ${t.contrast_pairs.map((p,i) => `
  <div style="background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--orange);border-radius:6px;padding:8px 10px;margin-bottom:6px">
    <div style="font-size:11px;color:var(--muted);margin-bottom:3px">#${i+1} — conf ${p.confidence!=null?(p.confidence*100).toFixed(0)+'%':'?'}</div>
    <div style="font-size:11px;color:var(--text)">"${esc((p.preview||'').substring(0,100))}…"</div>
  </div>`).join('')}
</div>` : ''}
`;
}

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const r = await fetch('/api/graph/data');
    const data = await r.json();
    buildGraph(data);
  } catch(e) {
    document.getElementById('graph-wrap').innerHTML =
      '<p style="color:var(--muted);padding:24px">Failed to load graph data</p>';
  }
}
init();
window.addEventListener('resize', () => {
  if(!sim) return;
  const {w, h} = getSize();
  sim.force('center', d3.forceCenter(w/2, h/2).strength(.04)).alpha(.3).restart();
});
</script>
</body>
</html>
"##;

// ── Chat HTML ─────────────────────────────────────────────────────────────────

pub async fn handle_chat_page() -> Html<&'static str> {
    Html(CHAT_HTML)
}


static CHAT_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ghost Chat</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #ACACAC;
  min-height: 100vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  font-family: 'Courier New', Courier, monospace;
}

.window {
  width: 100%;
  max-width: 880px;
  min-height: 100vh;
  background: #FFF;
  display: flex;
  flex-direction: column;
  box-shadow: 3px 3px 14px rgba(0,0,0,0.4);
}

/* Title bar */
.titlebar {
  background: linear-gradient(180deg, #3068C0 0%, #1A4DA0 100%);
  color: #FFF;
  height: 22px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 8px;
  flex-shrink: 0;
  position: relative;
  user-select: none;
}
.tb-btns { display: flex; gap: 5px; align-items: center; }
.tb-btn  { width: 11px; height: 11px; border-radius: 50%; border: 1px solid rgba(0,0,0,0.25); cursor: pointer; }
.tb-close { background: #FF5F57; }
.tb-min   { background: #FEBC2E; }
.tb-zoom  { background: #28C840; }
.tb-title {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 12px; font-weight: 600;
  position: absolute; left: 50%; transform: translateX(-50%);
  white-space: nowrap;
}
.tb-back { font-family: system-ui, sans-serif; font-size: 11px; color: #C0D4FF; text-decoration: none; }
.tb-back:hover { color: #FFF; }

/* Toolbar */
.toolbar {
  background: #DDE5F4;
  border-bottom: 1px solid #99AACC;
  padding: 3px 8px;
  display: flex; align-items: center; gap: 6px;
  flex-shrink: 0;
}
.t-btn {
  font-family: system-ui, sans-serif; font-size: 11px;
  background: linear-gradient(180deg, #F2F6FF 0%, #D2DFEF 100%);
  border: 1px solid #8899CC; border-radius: 3px;
  padding: 1px 10px; cursor: pointer; color: #112;
}
.t-btn:hover  { background: linear-gradient(180deg, #FAFCFF 0%, #DDEAF8 100%); }
.t-btn:active { filter: brightness(0.92); }
.t-sep    { color: #99AACC; }
.t-status { font-family: system-ui, sans-serif; font-size: 11px; color: #2255AA; }

/* Messages */
.msgs {
  flex: 1; overflow-y: auto;
  padding: 12px 14px; background: #FFF; min-height: 0;
}
.msgs::-webkit-scrollbar { width: 8px; }
.msgs::-webkit-scrollbar-track { background: #E8EEF8; }
.msgs::-webkit-scrollbar-thumb { background: #99AACC; }

.msg { margin-bottom: 14px; }

.msg-who {
  font-family: system-ui, sans-serif; font-size: 10px; font-weight: 700;
  letter-spacing: 0.07em; text-transform: uppercase;
  margin-bottom: 3px; display: flex; align-items: center; gap: 7px;
}
.msg-who.you { color: #334; }
.msg-who.gw  { color: #1A4DA0; }

.src-badge {
  font-size: 9px; font-weight: 400; letter-spacing: 0.04em;
  padding: 1px 5px; border-radius: 2px;
  border: 1px solid #99AACC; background: #EEF2FF; color: #3355AA;
}

.msg-body {
  font-size: 13px; line-height: 1.6;
  white-space: pre-wrap; word-break: break-word;
}
.msg-body.you { color: #223; }
.msg-body.gw  { color: #1A3D88; }

.msg-body pre {
  background: #EEF2FF; border: 1px solid #BCCADF; border-radius: 2px;
  padding: 8px 10px; margin: 6px 0; overflow-x: auto;
  font-size: 12px; color: #112; white-space: pre;
}
.msg-body code:not(pre code) {
  background: #EEF2FF; border: 1px solid #BCCADF; border-radius: 2px;
  padding: 1px 4px; font-size: 12px;
}
.msg-meta { font-family: system-ui, sans-serif; font-size: 10px; color: #7788BB; margin-top: 3px; }

/* Thinking indicator */
.thinking { font-family: system-ui, sans-serif; font-size: 12px; color: #6677AA; padding: 6px 0; display: none; }
.thinking.on { display: block; }

/* Status bar */
.statusbar {
  background: #DDE5F4; border-top: 1px solid #99AACC;
  padding: 2px 10px; flex-shrink: 0;
  font-family: system-ui, sans-serif; font-size: 10px; color: #334;
  display: flex; justify-content: space-between;
}

/* Input area */
.input-row {
  background: #EEF2FF; border-top: 1px solid #99AACC;
  padding: 7px 8px; display: flex; gap: 6px;
  align-items: flex-end; flex-shrink: 0;
}
.chat-in {
  flex: 1; border: 1px solid #7A90C8; border-radius: 2px;
  padding: 5px 7px; font-family: 'Courier New', Courier, monospace;
  font-size: 13px; color: #112; resize: none;
  min-height: 30px; max-height: 100px; outline: none;
  background: #FFF; line-height: 1.4; scrollbar-width: none;
}
.chat-in:focus    { border-color: #1A4DA0; }
.chat-in:disabled { background: #F0F4FA; color: #99A; }
.chat-in::placeholder { color: #99AACC; }

.send-btn {
  font-family: system-ui, sans-serif; font-size: 12px; font-weight: 700;
  color: #FFF;
  background: linear-gradient(180deg, #2D68C8 0%, #1A4DA0 100%);
  border: 1px solid #0F3A88; border-radius: 3px;
  padding: 0 18px; cursor: pointer; min-height: 30px; white-space: nowrap;
}
.send-btn:hover  { background: linear-gradient(180deg, #3E78D8 0%, #2A5EB0 100%); }
.send-btn:active { filter: brightness(0.9); }
.send-btn:disabled { opacity: 0.4; cursor: default; }
</style>
</head>
<body>

<div class="window">
  <div class="titlebar" style="position:relative">
    <div class="tb-btns">
      <div class="tb-btn tb-close" onclick="window.location.href='/'"></div>
      <div class="tb-btn tb-min"></div>
      <div class="tb-btn tb-zoom"></div>
    </div>
    <span class="tb-title">Ghost Chat</span>
    <a class="tb-back" href="/">&larr; Portal</a>
  </div>

  <div class="toolbar">
    <button class="t-btn" id="btn-clear">Clear</button>
    <span class="t-sep">|</span>
    <span class="t-status" id="t-status">Ready</span>
  </div>

  <div class="msgs" id="msgs">
    <div class="thinking" id="thinking">Ghost Chat is thinking&hellip;</div>
  </div>

  <div class="statusbar">
    <span id="sb-src">claude-cache</span>
    <span>Ghost Chat</span>
  </div>

  <div class="input-row">
    <textarea class="chat-in" id="chat-in" rows="1"
              placeholder="Type a message and press Enter to send&hellip;"
              autocomplete="off" spellcheck="false"></textarea>
    <button class="send-btn" id="send-btn">Send</button>
  </div>
</div>

<script>
var GW_SYSTEM = 'You are Ghost Chat, an assistant embedded in the claude-cache semantic proxy — a Rust caching layer that routes AI requests intelligently.\n\nThe proxy maintains a semantic SQLite cache, an embedding indexer, a local Ollama synthesis engine, and a federation mesh of peer nodes.\n\nRouting pipeline: exact cache hit → semantic cache → federation peers → local model → Anthropic API.\n\nBe direct and helpful. You know this architecture well and are a capable engineer. When writing code, prefer Rust.';

var SRC_LABELS = {
  'exact_cache':          'cache · exact',
  'semantic_cache':       'cache · semantic',
  'local':                'local model',
  'api':                  'api',
  'api-stream':           'api stream',
  'credit-bypass':        'bypass',
  'credit-bypass-stream': 'bypass stream',
};

var chatHistory = [];
var busy        = false;

function eid(id) { return document.getElementById(id); }
function scrollEnd() { var m = eid('msgs'); m.scrollTop = m.scrollHeight; }

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMd(raw) {
  var blocks = [];
  var t = raw.replace(/```(\w*)\n?([\s\S]*?)```/g, function(_, _lang, code) {
    var i = blocks.length;
    blocks.push('<pre><code>' + escHtml(code.trimEnd()) + '</code></pre>');
    return '\x00B' + i + '\x00';
  });
  t = escHtml(t)
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  return t.replace(/\x00B(\d+)\x00/g, function(_, i) { return blocks[+i]; });
}

function appendMsg(role, text, meta) {
  var msgs  = eid('msgs');
  var think = eid('thinking');

  var div = document.createElement('div');
  div.className = 'msg';

  var who = document.createElement('div');
  who.className = 'msg-who ' + (role === 'user' ? 'you' : 'gw');
  who.textContent = role === 'user' ? 'You' : 'Ghost Chat';

  if (meta && meta.src) {
    var badge = document.createElement('span');
    badge.className = 'src-badge';
    badge.textContent = SRC_LABELS[meta.src] || meta.src;
    who.appendChild(badge);
  }

  var body = document.createElement('div');
  body.className = 'msg-body ' + (role === 'user' ? 'you' : 'gw');
  if (role === 'user') {
    body.textContent = text;
  } else {
    body.innerHTML = renderMd(text);
  }

  div.appendChild(who);
  div.appendChild(body);

  if (meta && (meta.ms || meta.domain)) {
    var mt = document.createElement('div');
    mt.className = 'msg-meta';
    var parts = [];
    if (meta.ms)     { parts.push(meta.ms + ' ms'); }
    if (meta.domain) { parts.push(meta.domain + (meta.intent ? '/' + meta.intent : '')); }
    mt.textContent = parts.join(' · ');
    div.appendChild(mt);
  }

  msgs.insertBefore(div, think);
  scrollEnd();
}

function grow() {
  var ta = eid('chat-in');
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 100) + 'px';
}

function setStatus(s) { eid('t-status').textContent = s; }

async function send() {
  var inp  = eid('chat-in');
  var text = inp.value.trim();
  if (!text || busy) return;

  busy = true;
  eid('send-btn').disabled = true;
  inp.disabled = true;
  inp.value = '';
  grow();

  appendMsg('user', text, null);
  chatHistory.push({ role: 'user', content: text });

  eid('thinking').classList.add('on');
  scrollEnd();
  setStatus('Thinking…');

  var t0 = performance.now();
  try {
    var resp = await fetch('/v1/messages', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model:      'claude-sonnet-4-6',
        max_tokens: 2048,
        stream:     false,
        system:     GW_SYSTEM,
        messages:   chatHistory.map(function(m) {
          return { role: m.role, content: [{ type: 'text', text: m.content }] };
        }),
      }),
    });

    var ms = Math.round(performance.now() - t0);

    if (!resp.ok) {
      var errMsg = resp.status + ' ' + resp.statusText;
      try {
        var errData = await resp.json();
        if (errData.error && errData.error.message) { errMsg = errData.error.message; }
      } catch (_) {}
      throw new Error(errMsg);
    }

    var data  = await resp.json();
    var reply = (data.content || [])
      .filter(function(b) { return b.type === 'text'; })
      .map(function(b) { return b.text; })
      .join('\n');
    if (!reply) { reply = JSON.stringify(data, null, 2); }

    var src    = resp.headers.get('x-router-source') || 'api';
    var domain = resp.headers.get('x-cc-domain')     || '';
    var intent = resp.headers.get('x-cc-intent')     || '';

    chatHistory.push({ role: 'assistant', content: reply });
    appendMsg('ghost', reply, { src: src, ms: ms, domain: domain, intent: intent });
    eid('sb-src').textContent = SRC_LABELS[src] || src;
    setStatus('Ready');

  } catch (err) {
    appendMsg('ghost', '⚠ ' + err.message, null);
    setStatus('Error');
  } finally {
    eid('thinking').classList.remove('on');
    busy = false;
    eid('send-btn').disabled = false;
    inp.disabled = false;
    inp.focus();
  }
}

eid('chat-in').addEventListener('input', grow);
eid('chat-in').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
eid('send-btn').addEventListener('click', send);
eid('btn-clear').addEventListener('click', function() {
  if (busy) return;
  chatHistory.length = 0;
  var msgs  = eid('msgs');
  var think = eid('thinking');
  while (msgs.firstChild) { msgs.removeChild(msgs.firstChild); }
  msgs.appendChild(think);
  eid('sb-src').textContent = 'claude-cache';
  setStatus('Ready');
  eid('chat-in').focus();
});
</script>
</body>
</html>
"#;
