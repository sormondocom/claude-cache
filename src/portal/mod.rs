use axum::{extract::State, response::{Html, IntoResponse}, Json};
use serde_json::json;

use crate::server::SharedState;

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

    Json(json!({
        "node_id":       state.node_id,
        "federation": {
            "enabled":    state.federation.is_enabled(),
            "peer_count": state.federation.peer_count(),
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
  #toast { position:fixed; bottom:24px; right:24px; padding:10px 18px; border-radius:6px; font-size:.85rem; display:none; }
  #toast.ok  { background:var(--green); color:#000; }
  #toast.err { background:var(--red);   color:#fff; }
</style>
</head>
<body>
<h1>claude-cache</h1>
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
  <div class="card-label">Shared Cache Hashes (latest 50)</div>
  <div id="hashes" style="font-size:.7rem;color:var(--muted);margin-top:8px">loading...</div>
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
        </div>`).join('');

    const rows = (d.recent || []).slice(0, 25);
    if (!rows.length) {
      document.getElementById('routing-recent').innerHTML =
        '<p style="color:var(--muted);font-size:.85rem">no requests yet</p>';
      return;
    }
    document.getElementById('routing-recent').innerHTML = `
      <table style="margin-top:8px">
        <thead><tr>
          <th>Time</th><th>Shape</th><th>Decision</th><th>Latency</th><th>Tokens</th><th>Saved</th>
        </tr></thead>
        <tbody>
          ${rows.map(r => `<tr>
            <td style="color:var(--muted)">${timeAgo(r.created_at)}</td>
            <td style="font-size:.75rem">${r.shape_key || ''}</td>
            <td><span class="badge ${DECISION_BADGE[r.decision]||'badge-untrusted'}">${r.decision}</span></td>
            <td>${r.latency_ms} ms</td>
            <td style="color:var(--muted)">${r.tokens_in != null ? r.tokens_in : ''}</td>
            <td style="color:var(--green)">${r.saved_usd ? '$' + r.saved_usd.toFixed(4) : ''}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch(e) { console.error(e); }
}

async function refresh() {
  try {
    const ov = await fetch('/api/overview').then(r => r.json());
    document.getElementById('node-id').textContent = 'node: ' + ov.node_id;
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

  try {
    const ch = await fetch('/api/cache').then(r => r.json());
    const hashes = (ch.hashes || []).map(h => h.slice(0, 16) + '...').join('<br>');
    document.getElementById('hashes').innerHTML = hashes || '<em>none yet</em>';
  } catch(e) {}
}
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"#;
