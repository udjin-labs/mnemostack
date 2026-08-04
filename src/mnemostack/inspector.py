"""Web inspector — a memory operations & tenant-administration console.

A small, dependency-light operator UI for a mnemostack deployment. It has two
modes:

- **Read-only browse (default, no auth).** ``mnemostack inspect`` — look at what's
  stored: tenants, per-tenant collection/graph size, records (with filters),
  stale/invalidated facts, dependency reachability, and a smoke query. It never
  writes to the data, so it's safe to point at production. Data reads are always
  tenant-scoped by the vector store's boundary, so one tenant's view can't show
  another's records.

- **Admin console (``--auth``).** ``mnemostack inspect --auth`` requires an
  **admin**-scoped service key for every ``/api`` call and unlocks tenant
  administration: issue/revoke service keys and set/remove per-tenant quotas
  (storage + rate), on top of the browse views. Auth is by header
  (``X-API-Key`` / ``Authorization: Bearer``) supplied by the page's JS after the
  operator enters the key — never a cookie, so there is no CSRF surface. A freshly
  issued key's plaintext is shown **once** and never stored. Revoking the last
  admin key is refused, so the console can't lock itself out.

Run it separately from the serving API (it is an operator tool, not the public
recall surface). Bind to a trusted interface; even under ``--auth`` an admin
console warrants a localhost bind or a TLS-terminating proxy:

    mnemostack inspect --host 127.0.0.1 --port 8100          # read-only
    mnemostack inspect --auth --port 8100                    # admin console

Install the server extra: ``pip install 'mnemostack[server]'``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

try:
    from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "FastAPI is not installed. Install the optional server extra: "
        "`pip install 'mnemostack[server]'`."
    ) from e

from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mnemostack.auth import FileKeyStore

from mnemostack import __version__
from mnemostack.config import model_kwargs
from mnemostack.embeddings import get_provider
from mnemostack.embeddings.roles import embed_query_via, recall_space_error
from mnemostack.server import ServerConfig, _make_probe_client
from mnemostack.vector import VectorStore
from mnemostack.vector.qdrant import (
    TENANT_ID_KEY,
    _hide_invalidated_condition,
    _tenant_condition,
)

log = logging.getLogger(__name__)

# One self-contained page: no external scripts/styles/fonts (CSP-safe), plain and
# functional. Vanilla JS talks to the /api/* endpoints. Auth (admin console) is by
# header the JS sets from an in-memory key — never a cookie, so no CSRF surface.
INSPECTOR_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mnemostack inspector</title>
<style>
 :root{color-scheme:light dark}
 body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:1rem;max-width:1100px}
 h1{font-size:1.1rem;margin:.2rem 0 1rem}
 .muted{opacity:.65} .mono{font-family:ui-monospace,monospace}
 .bar{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem}
 select,input,button{font:inherit;padding:.35rem .5rem;border:1px solid #8886;border-radius:6px;background:transparent;color:inherit}
 input[type=text],input[type=password]{min-width:14rem}
 button{cursor:pointer}
 .cards{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem}
 .card{border:1px solid #8884;border-radius:8px;padding:.6rem .9rem;min-width:8rem}
 .card b{font-size:1.4rem;display:block}
 .ok{color:#2a8}.bad{color:#c44}.off{opacity:.5}
 table{border-collapse:collapse;width:100%}
 th,td{text-align:left;padding:.4rem .5rem;border-bottom:1px solid #8883;vertical-align:top}
 td.txt{max-width:44rem}
 tr.stale{opacity:.55}
 .pill{font-size:.75rem;border:1px solid #8886;border-radius:999px;padding:0 .5rem}
 pre{white-space:pre-wrap;word-break:break-word;background:#8881;padding:.6rem;border-radius:6px}
 nav.tabs{display:flex;gap:.3rem;margin:.4rem 0 1rem;border-bottom:1px solid #8883}
 nav.tabs button{border:0;border-bottom:2px solid transparent;border-radius:0;background:transparent;padding:.4rem .8rem}
 nav.tabs button.active{border-bottom-color:currentColor;font-weight:600}
 .panel{border:1px solid #8884;border-radius:8px;padding:.8rem;margin-bottom:1rem}
 .plaintext{background:#2a81;border:1px solid #2a8;border-radius:6px;padding:.6rem;margin:.6rem 0}
 .err{color:#c44}
 label.chk{border:0;padding:0;display:inline-flex;gap:.2rem;align-items:center}
</style></head><body>
<h1>mnemostack inspector <span class="muted mono" id="ver"></span> <span class="pill" id="mode">read-only</span></h1>
<div class="bar" id="keybar" hidden>
 <b>admin key required</b>
 <input type="password" id="key" placeholder="admin service key (msk_…)" autocomplete="off">
 <button id="keygo">Unlock</button>
 <span class="err" id="keyerr"></span>
</div>
<nav class="tabs">
 <button data-tab="browse" class="active">Browse</button>
 <button data-tab="keys" hidden>Keys</button>
 <button data-tab="quotas" hidden>Quotas</button>
 <button data-tab="audit" hidden>Audit</button>
</nav>

<section id="tab-browse">
 <div class="bar">
  <label>tenant <select id="tenant"></select></label> <input id="tenant-manual" placeholder="or type a tenant id" size="16">
  <input type="text" id="q" placeholder="smoke query (vector search) — empty = browse records">
  <input type="text" id="filters" placeholder='filters JSON e.g. {"source":"notes.md"}' style="min-width:20rem">
  <button id="go">Search</button>
  <span class="muted" id="status"></span>
 </div>
 <div class="cards" id="cards"></div>
 <table><thead><tr><th>id</th><th>source</th><th>text</th><th>state</th></tr></thead>
 <tbody id="rows"></tbody></table>
 <pre id="detail" hidden></pre>
</section>

<section id="tab-keys" hidden>
 <div class="panel">
  <b>Issue a service key</b>
  <div class="bar">
   <input type="text" id="k-tenant" placeholder="tenant id">
   <label class="chk"><input type="checkbox" class="k-scope" value="read" checked>read</label>
   <label class="chk"><input type="checkbox" class="k-scope" value="write">write</label>
   <label class="chk"><input type="checkbox" class="k-scope" value="admin">admin</label>
   <input type="text" id="k-label" placeholder="label (optional)">
   <button id="k-issue">Issue</button>
   <span class="err" id="k-err"></span>
  </div>
  <div class="muted">admin is a <b>global</b> operator scope — full access to every tenant's
   keys and quotas, not just the tenant above.</div>
  <div class="plaintext" id="k-plain" hidden></div>
 </div>
 <table><thead><tr><th>id</th><th>tenant</th><th>scopes</th><th>label</th><th>created</th><th></th></tr></thead>
 <tbody id="k-rows"></tbody></table>
</section>

<section id="tab-quotas" hidden>
 <div class="panel">
  <b>Set a tenant quota</b> <span class="muted">(blank = leave unchanged; sets only what you fill)</span>
  <div class="bar">
   <input type="text" id="q-tenant" placeholder="tenant id">
   <input type="text" id="q-points" placeholder="max_points" size="10">
   <input type="text" id="q-rps" placeholder="max_rps" size="8">
   <input type="text" id="q-burst" placeholder="burst" size="6">
   <button id="q-set">Set</button>
   <span class="err" id="q-err"></span>
  </div>
 </div>
 <table><thead><tr><th>tenant</th><th>max_points</th><th>max_rps</th><th>burst</th><th></th></tr></thead>
 <tbody id="q-rows"></tbody></table>
</section>

<section id="tab-audit" hidden>
 <div class="bar">
  <button id="a-refresh">Refresh</button>
  <span class="muted" id="a-status"></span>
 </div>
 <table><thead><tr><th>time</th><th>action</th><th>actor</th><th>tenant</th><th>outcome</th><th>details</th></tr></thead>
 <tbody id="a-rows"></tbody></table>
</section>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=t=>String(t??"").replace(/[<&>"']/g,c=>({"<":"&lt;","&":"&amp;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let KEY="";  // admin key, in-memory only (never persisted) — lost on reload by design
async function j(u,opts){
  opts=opts||{}; opts.headers=Object.assign({},opts.headers);
  if(KEY) opts.headers["X-API-Key"]=KEY;
  if(opts.body){opts.headers["Content-Type"]="application/json";}
  const r=await fetch(u,opts);
  const body=await r.text();
  if(!r.ok){let d;try{d=JSON.parse(body).detail}catch(e){d=body}
    const err=new Error((d||r.statusText)); err.status=r.status; throw err;}
  return body?JSON.parse(body):{};
}
function tenant(){return $("#tenant-manual").value.trim()||$("#tenant").value;}
function effBurst(q){return q.burst!=null?q.burst:(q.max_rps!=null?Math.max(1,Math.ceil(q.max_rps)):null);}

// ---- tabs ----
$$("nav.tabs button").forEach(b=>b.addEventListener("click",()=>{
  $$("nav.tabs button").forEach(x=>x.classList.remove("active"));b.classList.add("active");
  for(const t of ["browse","keys","quotas","audit"])$("#tab-"+t).hidden=(t!==b.dataset.tab);
}));

// ---- browse ----
async function loadTenants(){
  const d=await j("/api/tenants?limit=1000");$("#ver").textContent="v"+d.version;
  const sel=$("#tenant");sel.innerHTML='<option value="">(all / unscoped)</option>';
  if(d.ok===false){$("#status").textContent=d.error||"tenant lookup failed";}
  for(const t of d.tenants){const o=document.createElement("option");o.value=t.id;o.textContent=t.id+" ("+t.count+")";sel.appendChild(o);}
}
async function loadOverview(){
  const d=await j("/api/overview?tenant="+encodeURIComponent(tenant()));
  const dep=(ok,label)=>`<div class="card"><b class="${ok===null?'off':ok?'ok':'bad'}">${ok===null?'—':ok?'up':'down'}</b>${label}</div>`;
  $("#cards").innerHTML=
    `<div class="card"><b>${d.points}</b>points</div>`+
    `<div class="card"><b>${d.invalidated}</b>invalidated</div>`+
    dep(d.qdrant,"qdrant")+dep(d.memgraph,"memgraph (graph)");
}
async function loadRecords(){
  $("#detail").hidden=true;$("#detail").textContent="";$("#status").textContent="loading…";
  const p=new URLSearchParams({tenant:tenant(),limit:"50"});
  if($("#q").value)p.set("q",$("#q").value);
  if($("#filters").value)p.set("filters",$("#filters").value);
  let d;try{d=await j("/api/records?"+p);}catch(e){$("#status").textContent=e.message;return;}
  if(d.error){$("#status").textContent=d.error;$("#rows").innerHTML="";window._recs={};return;}
  $("#status").textContent=d.records.length+" record(s)"+(d.mode?" · "+d.mode:"");
  $("#rows").innerHTML=d.records.map(r=>
    `<tr class="${r.invalidated?'stale':''}" data-id="${esc(r.id)}">
      <td class="mono">${esc(r.id).slice(0,12)}…</td><td>${esc(r.source)}</td>
      <td class="txt">${esc(r.text).slice(0,300)}</td>
      <td>${r.invalidated?'<span class="pill">stale</span>':''}</td></tr>`).join("");
  window._recs=Object.fromEntries(d.records.map(r=>[r.id,r]));
}
$("#rows").addEventListener("click",e=>{const tr=e.target.closest("tr");if(!tr)return;
  const r=window._recs[tr.dataset.id];if(!r)return;
  $("#detail").hidden=false;$("#detail").textContent=JSON.stringify(r,null,2);});
$("#tenant").addEventListener("change",()=>{loadOverview();loadRecords();});
$("#tenant-manual").addEventListener("change",()=>{loadOverview();loadRecords();});
$("#go").addEventListener("click",loadRecords);
$("#q").addEventListener("keydown",e=>{if(e.key==="Enter")loadRecords();});

// ---- keys management ----
async function loadKeys(){
  const d=await j("/api/keys");
  $("#k-rows").innerHTML=d.keys.map(k=>
    `<tr><td class="mono">${esc(k.id)}</td><td>${esc(k.tenant)}</td>
      <td>${(k.scopes||[]).map(s=>'<span class="pill">'+esc(s)+'</span>').join(" ")}</td>
      <td>${esc(k.label||"")}</td><td class="muted">${esc(k.created_at||"")}</td>
      <td><button data-revoke="${esc(k.id)}">revoke</button></td></tr>`).join("")
    ||'<tr><td colspan="6" class="muted">no keys</td></tr>';
}
$("#k-issue").addEventListener("click",async()=>{
  $("#k-err").textContent="";$("#k-plain").hidden=true;
  const scopes=$$(".k-scope").filter(c=>c.checked).map(c=>c.value);
  if(scopes.includes("admin")&&!confirm(
    "An admin key has FULL access to ALL tenants (their keys and quotas), not just "
    +"the tenant you entered. Issue this admin key?"))return;
  try{
    const d=await j("/api/keys",{method:"POST",body:JSON.stringify(
      {tenant:$("#k-tenant").value.trim(),scopes,label:$("#k-label").value})});
    $("#k-plain").hidden=false;
    $("#k-plain").innerHTML="key <b class='mono'>"+esc(d.key)+"</b> for <b>"+esc(d.tenant)+
      "</b> — <b>copy now, it is not shown again</b>";
    $("#k-tenant").value="";$("#k-label").value="";await loadKeys();
  }catch(e){$("#k-err").textContent=e.message;}
});
$("#k-rows").addEventListener("click",async e=>{
  const id=e.target.dataset.revoke;if(!id)return;
  if(!confirm("Revoke key "+id+"?"))return;
  try{await j("/api/keys/"+encodeURIComponent(id),{method:"DELETE"});await loadKeys();}
  catch(err){alert(err.message);}
});

// ---- quotas management ----
async function loadQuotas(){
  const d=await j("/api/quotas");
  const fmt=v=>v==null?'<span class="muted">—</span>':esc(v);
  $("#q-rows").innerHTML=d.quotas.map(q=>
    `<tr><td>${esc(q.tenant)}</td><td>${fmt(q.max_points)}</td><td>${fmt(q.max_rps)}</td>
      <td>${fmt(effBurst(q))}</td>
      <td><button data-rmq="${esc(q.tenant)}">remove</button></td></tr>`).join("")
    ||'<tr><td colspan="5" class="muted">no quotas</td></tr>';
}
$("#q-set").addEventListener("click",async()=>{
  $("#q-err").textContent="";
  const t=$("#q-tenant").value.trim();if(!t){$("#q-err").textContent="tenant required";return;}
  // Only include a field the operator actually typed (partial update). Validate
  // numerics client-side: a non-numeric typo must be an error, NOT silently sent
  // as null (which the server would read as "clear this cap").
  const body={};
  for(const [id,key,int] of [["#q-points","max_points",1],["#q-rps","max_rps",0],["#q-burst","burst",1]]){
    const s=$(id).value.trim(); if(s==="")continue;
    const re=int?/^\\d+$/:/^\\d*\\.?\\d+$/;
    if(!re.test(s)){$("#q-err").textContent=key+(int?" must be a whole number":" must be a number");return;}
    body[key]=int?parseInt(s,10):parseFloat(s);
  }
  if(!Object.keys(body).length){$("#q-err").textContent="fill at least one field";return;}
  try{await j("/api/quotas/"+encodeURIComponent(t),{method:"PUT",body:JSON.stringify(body)});
    $("#q-points").value=$("#q-rps").value=$("#q-burst").value="";await loadQuotas();}
  catch(e){$("#q-err").textContent=e.message;}
});
$("#q-rows").addEventListener("click",async e=>{
  const t=e.target.dataset.rmq;if(!t)return;
  if(!confirm("Remove quota for "+t+"?"))return;
  try{await j("/api/quotas/"+encodeURIComponent(t),{method:"DELETE"});await loadQuotas();}
  catch(err){alert(err.message);}
});

// ---- audit ----
async function loadAudit(){
  const d=await j("/api/audit");
  if(!d.enabled){$("#a-status").textContent="auditing is off — set MNEMOSTACK_AUDIT_FILE and restart";$("#a-rows").innerHTML="";return;}
  $("#a-status").textContent=d.skipped?d.skipped+" unparseable line(s) skipped":"";
  $("#a-rows").innerHTML=d.events.slice().reverse().map(e=>
    `<tr><td>${esc((e.ts||"").replace("T"," ").slice(0,19))}</td><td>${esc(e.action)}</td>`+
    `<td>${esc(e.actor)}</td><td>${esc(e.tenant??"")}</td><td>${esc(e.outcome)}</td>`+
    `<td class="muted">${esc(JSON.stringify(e.details||{}))}</td></tr>`).join("");
}
$("#a-refresh").addEventListener("click",()=>loadAudit().catch(e=>{$("#a-status").textContent=e.message;}));

// ---- boot / auth ----
async function loadBrowse(){await loadTenants();await loadOverview();await loadRecords();}
async function probeManage(){
  // /api/keys is admin-only; 200 => admin console, 403 => read-only/no-auth mode,
  // 501 => admin console but keys are managed externally (verify-only backend,
  // e.g. OpenBao) — hide the Keys panel, keep quotas manageable.
  try{await loadKeys();$('nav.tabs button[data-tab="keys"]').hidden=false;}
  catch(e){
    if(e.status===403)return;          // not an admin console; stay read-only
    if(e.status!==501)throw e;         // 401 -> boot() keybar; others surface
  }
  $("#mode").textContent="admin";
  // Quotas independently: a broken/unreadable quota store (503) must not hide the
  // otherwise-usable keys panel — and vice versa (external keys, local quotas).
  try{await loadQuotas();$('nav.tabs button[data-tab="quotas"]').hidden=false;}
  catch(e){/* leave the quotas tab hidden; the rest of the console still works */}
  // Audit tab is ALWAYS shown in admin mode: a configured-but-unreadable trail
  // must surface as a visible error (the read fails loud by design), not a
  // silently missing tab that masks the outage. Disabled auditing renders its
  // own how-to-enable notice inside the tab.
  try{await loadAudit();}
  catch(e){$("#a-status").textContent="audit log unreadable: "+e.message;$("#a-rows").innerHTML="";}
  $('nav.tabs button[data-tab="audit"]').hidden=false;
}
async function boot(){
  try{await loadBrowse();$("#keybar").hidden=true;await probeManage();}
  catch(e){
    if(e.status===401){$("#keybar").hidden=false;$("#mode").textContent="auth required";
      if(KEY)$("#keyerr").textContent="invalid or non-admin key";}
    else $("#status").textContent=e.message;
  }
}
$("#keygo").addEventListener("click",()=>{KEY=$("#key").value.trim();$("#keyerr").textContent="";boot();});
$("#key").addEventListener("keydown",e=>{if(e.key==="Enter")$("#keygo").click();});
boot();
</script></body></html>"""


def _graph_reachable(cfg: ServerConfig) -> bool | None:
    """Ping the graph if configured. None = not configured (not an error)."""
    if not cfg.graph_uri:
        return None
    try:
        from neo4j import GraphDatabase
    except Exception:  # noqa: BLE001 — neo4j not installed
        return False
    driver = None
    try:
        driver = GraphDatabase.driver(
            cfg.graph_uri,
            auth=(cfg.graph_user, cfg.graph_password) if cfg.graph_user else None,
            connection_timeout=cfg.graph_health_timeout,
            connection_acquisition_timeout=cfg.graph_health_timeout,
        )
        with driver.session(database=cfg.graph_database) as s:
            s.run("RETURN 1").single()
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        # /api/overview probes on every page load / tenant switch, so always close
        # the driver — a persistent graph failure must not leak driver pools.
        if driver is not None:
            driver.close()


def _legacy_only_filter(store: Any, parsed: dict[str, Any] | None) -> Filter:
    """Filter for legacy/unscoped points — those with NO ``tenant_id`` — AND-combined
    with any user filters. The inspector's unscoped mode uses this so it shows only
    untenanted points (a default single-tenant collection), never another tenant's
    data — upholding the never-cross-tenant contract on a mixed collection too.
    """
    must = list(store._build_filter(parsed).must or []) if parsed else []
    must.append(IsEmptyCondition(is_empty=PayloadField(key=TENANT_ID_KEY)))
    return Filter(must=must)


class _KeyCreate(BaseModel):
    tenant: str = Field(..., min_length=1, max_length=200)
    scopes: list[str] = Field(..., min_length=1)
    label: str = Field("", max_length=200)


def build_inspector_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or ServerConfig.from_env()
    # Admin console (issue/revoke keys, manage quotas) is unlocked by `--auth`,
    # which then requires an admin key on every /api call. Without --auth the
    # management stores aren't even opened — the console is a read-only browser.
    key_store = None
    quota_store = None
    if cfg.auth_enabled:
        from mnemostack.auth import make_key_store
        from mnemostack.quotas import FileQuotaStore

        # Backend selected by MNEMOSTACK_KEYSTORE. An external (e.g. OpenBao)
        # store is verify-only: auth works, but key MANAGEMENT stays in the
        # store's own tooling — the /api/keys endpoints answer 501 and the UI
        # hides the Keys panel (quotas remain manageable either way).
        key_store = make_key_store(cfg.keys_file)
        quota_store = FileQuotaStore(cfg.quotas_file)
    keys_manageable = all(
        hasattr(key_store, m) for m in ("issue", "revoke_guarded", "list_keys")
    )
    # Browse views (tenants / overview / records scroll) use count / scroll / facet
    # and need no embeddings, so don't construct the provider eagerly — that would
    # make `mnemostack inspect` require GEMINI_API_KEY (or the HF model/deps) just to
    # browse read-only Qdrant data. Build the provider lazily, only when a `?q=`
    # search must embed. The store's dimension is unused for browse/search here
    # (search is handed a precomputed vector), so a placeholder value is fine.
    store = VectorStore(collection=cfg.collection, dimension=1, host=cfg.qdrant_url)
    # A separate short-timeout client for reachability probes, so a slow or
    # blackholed Qdrant shows as "down" promptly instead of hanging the console for
    # the store's full (30s) client timeout.
    probe_client = _make_probe_client(cfg.qdrant_url, cfg.qdrant_health_timeout)
    _provider: dict[str, Any] = {}

    def _get_provider() -> Any:
        if "p" not in _provider:
            _provider["p"] = get_provider(cfg.provider_name, **model_kwargs(cfg.embedding_model))
        return _provider["p"]

    def _embed(text: str) -> list[float]:
        # Operator-typed search text is a retrieval query.
        return embed_query_via(_get_provider(), text)

    _space_verdict: dict[str, str | None] = {}

    def _space_error_cached() -> str | None:
        # The inspector searches the store directly (no Recaller), so it
        # needs its own embedding-space guard — silently misleading results
        # in an operator debugging tool are worse than an error. Conclusive
        # verdicts are cached per process; an inconclusive check (store
        # hiccup) lets the search proceed and is retried next time.
        if "verdict" in _space_verdict:
            return _space_verdict["verdict"]
        try:
            verdict = recall_space_error(store, _get_provider())
        except Exception:  # noqa: BLE001 — inconclusive must not block the tool
            return None
        _space_verdict["verdict"] = verdict
        return verdict

    app = FastAPI(title="mnemostack inspector", version=__version__)

    def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
        if x_api_key:
            return x_api_key.strip()
        if authorization and authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return None

    def _audit_evt(
        action: str,
        *,
        principal: Any = None,
        actor: str | None = None,
        tenant: str | None = None,
        outcome: str = "success",
        **details: Any,
    ) -> None:
        """Best-effort audit of a console action (no-op unless
        ``MNEMOSTACK_AUDIT_FILE`` is set; never raises — module contract).
        The actor is the admin key's public id — never key material."""
        from mnemostack.audit import audit_log_from_env

        if actor is None:
            if principal is not None and getattr(principal, "key_id", None):
                actor = f"key:{principal.key_id}"
            elif principal is not None:
                # A backend whose records carry no id (e.g. a bare OpenBao
                # record) — attribute to the key's tenant, the next-best handle.
                actor = f"key:?(tenant={principal.tenant})"
            else:
                actor = "anonymous"
        audit_log_from_env().record(
            action,
            tenant=tenant,
            actor=actor,
            surface="inspector",
            outcome=outcome,
            details=details or None,
        )

    def _require(*, manage: bool):
        """Dependency: admin key required. ``manage`` endpoints (keys/quotas) also
        require ``--auth`` to be on at all — they're never reachable unauthenticated.
        Browse endpoints are open in the default (no-auth) read-only mode."""

        def _dep(
            request: Request,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        ):
            if not cfg.auth_enabled:
                if manage:
                    raise HTTPException(
                        status_code=403,
                        detail="key/quota management requires `mnemostack inspect --auth`",
                    )
                return None  # read-only browse is open without --auth (legacy)
            key = _extract_key(authorization, x_api_key)
            if not key:
                # NOT audited: a missing key is an anonymous probe (every page
                # load fires them) — that's access-log noise, not a security
                # signal. Presented-but-rejected credentials below ARE logged.
                raise HTTPException(status_code=401, detail="missing admin key")
            client_ip = request.client.host if request.client else "?"
            principal = key_store.verify(key) if key_store else None
            if principal is None:
                _audit_evt(
                    "auth.denied",
                    actor=f"ip:{client_ip}",
                    outcome="denied",
                    reason="invalid_key",
                    path=request.url.path,
                )
                raise HTTPException(status_code=401, detail="invalid admin key")
            if not principal.can("admin"):
                _audit_evt(
                    "auth.denied",
                    principal=principal,
                    tenant=principal.tenant,  # known here, unlike invalid_key
                    outcome="denied",
                    reason="not_admin",
                    path=request.url.path,
                )
                raise HTTPException(status_code=403, detail="admin scope required")
            return principal

        return _dep

    _browse = Depends(_require(manage=False))
    _admin = Depends(_require(manage=True))

    def _qdrant_ok() -> bool:
        try:
            probe_client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _tenants_via_scroll(cap: int = 50000) -> list[dict[str, Any]]:
        """Distinct tenant_id values by scrolling payloads — the fallback for a
        Qdrant server older than the Facet API (1.12). Bounded scan (approximate
        for a very large corpus)."""
        counts: dict[Any, int] = {}
        offset: Any = None
        scanned = 0
        while scanned < cap:
            points, offset = store.client.scroll(
                collection_name=cfg.collection,
                with_payload=[TENANT_ID_KEY],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for p in points:
                tid = (p.payload or {}).get(TENANT_ID_KEY)
                if tid is not None:
                    counts[tid] = counts.get(tid, 0) + 1
            scanned += len(points)
            if offset is None or not points:
                break
        return [{"id": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))]

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INSPECTOR_HTML

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    def _admin_tenants() -> set[str]:
        """Tenants known to the operator's config (key store ∪ quota store), so a
        tenant that has a key or a quota but no data yet still appears in the list.
        Empty (and silent) in read-only mode or if a store is unreadable."""
        names: set[str] = set()
        if key_store is not None and keys_manageable:
            # A verify-only external store can't enumerate keys — quota tenants
            # (below) and the data facet still populate the list.
            try:
                lister = cast("FileKeyStore", key_store)  # duck-checked above
                names.update(k["tenant"] for k in lister.list_keys())
            except Exception as e:  # noqa: BLE001 — never fail the tenant list on this
                log.info("key store tenant merge failed: %s", e)
        if quota_store is not None:
            try:
                names.update(q["tenant"] for q in quota_store.list_quotas())
            except Exception as e:  # noqa: BLE001
                log.info("quota store tenant merge failed: %s", e)
        return names

    def _merge_config_tenants(data_tenants: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Union the data-derived tenants with config tenants (count 0 for the latter)."""
        extra = _admin_tenants() - {str(t["id"]) for t in data_tenants}
        merged = list(data_tenants) + [{"id": t, "count": 0} for t in extra]
        return sorted(merged, key=lambda t: str(t["id"]))

    @app.get("/api/tenants")
    def tenants(limit: int = Query(200, ge=1, le=1000), _p=_browse) -> dict[str, Any]:
        """Distinct tenants with per-tenant point counts.

        Under ``--auth`` the list is the union of tenants seen in the data and
        tenants known to the key/quota stores (the latter with count 0), so a
        provisioned-but-empty tenant is visible. Facet-based for the data side —
        cheap and approximate for very large corpora.
        """
        # Short-timeout reachability first: the data client (facet below) has a 30s
        # timeout, so a blackholed Qdrant would otherwise hang the initial load.
        if not _qdrant_ok():
            # Qdrant down, but config tenants (keys/quotas) are still worth showing.
            cfg_only = [{"id": t, "count": 0} for t in sorted(_admin_tenants())]
            return {
                "tenants": cfg_only,
                "ok": bool(cfg_only),
                "error": "Qdrant unreachable",
                "version": __version__,
            }
        try:
            resp = store.client.facet(
                collection_name=cfg.collection, key=TENANT_ID_KEY, limit=limit
            )
            out = [{"id": h.value, "count": h.count} for h in resp.hits]
            return {"tenants": _merge_config_tenants(out), "ok": True, "version": __version__}
        except Exception as e:  # noqa: BLE001
            # Facet failed. If Qdrant is unreachable, surface that (don't render as
            # "no tenants"). If it's reachable, the Facet API is likely just missing
            # (server < 1.12) — fall back to a bounded scroll scan so operators on
            # older Qdrant can still browse.
            if not _qdrant_ok():
                log.info("tenant facet failed, qdrant unreachable: %s", e)
                return {"tenants": [], "ok": False, "error": "Qdrant unreachable", "version": __version__}
            log.info("tenant facet unavailable, falling back to scroll: %s", e)
            try:
                return {
                    "tenants": _merge_config_tenants(_tenants_via_scroll()),
                    "ok": True,
                    "scanned": True,
                    "version": __version__,
                }
            except Exception as e2:  # noqa: BLE001
                log.info("tenant scroll discovery failed: %s", e2)
                return {
                    "tenants": [],
                    "ok": False,
                    "error": f"tenant discovery failed: {e2}",
                    "version": __version__,
                }

    @app.get("/api/overview")
    def overview(tenant: str = Query(""), _p=_browse) -> dict[str, Any]:
        # tenant="" browses unscoped: all points, including a legacy single-tenant
        # collection whose points carry no tenant_id. A named tenant scopes counts.
        scoped = tenant or None
        ok = _qdrant_ok()  # short-timeout probe up front so a down Qdrant is prompt
        points = 0
        invalidated = 0
        if ok:
            try:
                if scoped:
                    points = store.count(tenant=scoped)
                    inv_must: list[Any] = [_tenant_condition(scoped)]
                else:
                    # unscoped = legacy points only (no tenant_id), never cross-tenant
                    points = store.client.count(
                        collection_name=cfg.collection,
                        count_filter=_legacy_only_filter(store, None),
                    ).count
                    inv_must = [IsEmptyCondition(is_empty=PayloadField(key=TENANT_ID_KEY))]
                # points that DO carry an invalidated_at marker (must_not
                # "is-current" == has invalidated_at), within the same scope.
                invalidated = store.client.count(
                    collection_name=cfg.collection,
                    count_filter=Filter(must=inv_must, must_not=[_hide_invalidated_condition()]),
                ).count
            except Exception as e:  # noqa: BLE001
                log.warning("overview count failed for tenant=%r: %s", tenant, e)
        return {
            "tenant": tenant,
            "collection": cfg.collection,
            "points": points,
            "invalidated": invalidated,
            "qdrant": ok,
            "memgraph": _graph_reachable(cfg),
            "version": __version__,
        }

    @app.get("/api/records")
    def records(
        tenant: str = Query(""),
        q: str | None = Query(None),
        filters: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
        _p=_browse,
    ) -> dict[str, Any]:
        """Records for a tenant. `tenant=""` browses unscoped (all points, incl. a
        legacy single-tenant collection with no tenant_id); a named tenant scopes
        the read. With `q`, a vector smoke search; otherwise a browse (scroll)."""
        import json

        # Short-timeout reachability first so a blackholed Qdrant fails promptly
        # instead of hanging on the data client's 30s timeout.
        if not _qdrant_ok():
            return {"records": [], "error": "Qdrant unreachable", "tenant": tenant}

        scoped = tenant or None
        parsed: dict[str, Any] | None = None
        if filters:
            try:
                parsed = json.loads(filters)
                if not isinstance(parsed, dict):
                    return {"records": [], "error": "filters must be a JSON object"}
            except json.JSONDecodeError as e:
                return {"records": [], "error": f"invalid filters JSON: {e}"}
            # An operator's timestamp condition must reach Qdrant in the
            # collection's own domain (an ISO range over a numeric field
            # matches nothing) — one conversion covers every branch below,
            # including the legacy-only filter.
            from mnemostack.recall.retrievers import convert_timestamp_filter

            parsed = convert_timestamp_filter(
                parsed,
                timestamp_key=cfg.timestamp_key,
                timestamp_format=cfg.timestamp_format,
            )

        rows: list[dict[str, Any]] = []
        try:
            if q:
                space_err = _space_error_cached()
                if space_err:
                    return {"records": [], "error": space_err, "tenant": tenant}
                vec = _embed(q)
                if scoped:
                    for hit in store.search(vec, limit=limit, filters=parsed, tenant=scoped):
                        rows.append(_row(hit.id, hit.payload, score=hit.score, text_key=cfg.text_key))
                else:  # unscoped = legacy points only (no tenant_id)
                    res = store.client.query_points(
                        collection_name=cfg.collection,
                        query=vec,
                        limit=limit,
                        query_filter=_legacy_only_filter(store, parsed),
                        with_payload=True,
                    )
                    for pt in res.points:
                        rows.append(_row(pt.id, pt.payload or {}, score=pt.score, text_key=cfg.text_key))
                mode = "vector search"
            else:
                if scoped:
                    for hit in store.scroll(filters=parsed, tenant=scoped):
                        rows.append(_row(hit.id, hit.payload, text_key=cfg.text_key))
                        if len(rows) >= limit:
                            break
                else:  # unscoped browse: legacy points only
                    points, _ = store.client.scroll(
                        collection_name=cfg.collection,
                        scroll_filter=_legacy_only_filter(store, parsed),
                        with_payload=True,
                        limit=limit,
                    )
                    for rec in points:
                        rows.append(_row(rec.id, rec.payload or {}, text_key=cfg.text_key))
                mode = "browse"
        except Exception as e:  # noqa: BLE001 — a malformed filter/embed error is a
            # user/runtime problem, not a server bug: return a clean message like
            # the JSON-parse path, never a raw 500 (mirrors /api/overview).
            log.info("records query failed for tenant=%r: %s", tenant, e)
            return {"records": [], "error": f"query failed: {e}", "tenant": tenant}
        return {"records": rows, "mode": mode, "tenant": tenant}

    # ----- Admin console: service keys + quotas (--auth only, admin scope) -----

    def _require_manageable_keys() -> FileKeyStore:
        """The manageable key store, or 501 when the selected backend is
        verify-only (e.g. OpenBao): the console must not pretend to manage keys
        the servers don't verify against — lifecycle belongs to the external
        store's own tooling. Returns the store typed with the management surface
        (duck-checked at app build via ``keys_manageable``)."""
        if key_store is None or not keys_manageable:
            raise HTTPException(
                status_code=501,
                detail=(
                    "service keys are managed externally "
                    "(MNEMOSTACK_KEYSTORE selects a verify-only backend); "
                    "use the key store's own tooling"
                ),
            )
        return cast("FileKeyStore", key_store)

    @app.get("/api/keys")
    def list_keys(_p=_admin) -> dict[str, Any]:
        """Service keys (redacted — id/tenant/scopes/label/created; never a plaintext
        or hash). The key store fails closed, so a broken store surfaces as an error."""
        from mnemostack.auth import KeyStoreError

        ks = _require_manageable_keys()
        try:
            return {"keys": ks.list_keys()}
        except KeyStoreError as e:
            raise HTTPException(status_code=503, detail=f"key store unreadable: {e}") from e

    @app.post("/api/keys", status_code=201)
    def create_key(body: _KeyCreate, _p=_admin) -> dict[str, Any]:
        """Issue a service key. The plaintext is returned ONCE (never stored) — the
        caller must copy it now."""
        from mnemostack.auth import SCOPES, KeyStoreError

        ks = _require_manageable_keys()
        bad = [s for s in body.scopes if s not in SCOPES]
        if bad:
            raise HTTPException(
                status_code=400,
                detail=f"unknown scope(s) {bad}; valid: {sorted(SCOPES)}",
            )
        try:
            key_id, plaintext = ks.issue(
                body.tenant.strip(), body.scopes, label=body.label
            )
        except ValueError as e:  # bad tenant/scope input → client error
            raise HTTPException(status_code=400, detail=str(e)) from e
        except KeyStoreError as e:  # unreadable/corrupt store → server error (like the rest)
            _audit_evt(
                "keys.issue",
                principal=_p,
                tenant=body.tenant.strip(),
                outcome="error",
                error=str(e),
            )
            raise HTTPException(status_code=503, detail=f"key store error: {e}") from e
        # key_id only — never the plaintext or its hash (audit module contract).
        _audit_evt(
            "keys.issue",
            principal=_p,
            tenant=body.tenant.strip(),
            key_id=key_id,
            scopes=body.scopes,
            label=body.label,
        )
        return {"id": key_id, "key": plaintext, "tenant": body.tenant.strip(),
                "scopes": body.scopes, "label": body.label}

    @app.delete("/api/keys/{key_id}")
    def revoke_key(key_id: str, _p=_admin) -> dict[str, Any]:
        """Revoke a key. Refuses to revoke the LAST admin key (atomically) so the
        console can't lock itself (and every other admin) out."""
        from mnemostack.auth import KeyStoreError

        ks = _require_manageable_keys()
        # Attribution lookup BEFORE deletion (best-effort — a lookup failure
        # must never block the revocation), so the event says whose credential
        # was removed without joining to a possibly-rotated-out issue event.
        key_tenant: str | None = None
        try:
            key_tenant = next(
                (k.get("tenant") for k in ks.list_keys() if k.get("id") == key_id), None
            )
        except Exception:  # noqa: BLE001 — attribution only; revoke proceeds
            pass
        try:
            status = ks.revoke_guarded(key_id, protect_last_admin=True)
        except KeyStoreError as e:
            _audit_evt(
                "keys.revoke",
                principal=_p,
                tenant=key_tenant,
                outcome="error",
                key_id=key_id,
                error=str(e),
            )
            raise HTTPException(status_code=503, detail=f"key store error: {e}") from e
        if status == "not_found":
            _audit_evt(
                "keys.revoke",
                principal=_p,
                tenant=key_tenant,
                outcome="error",
                key_id=key_id,
                reason="not_found",
            )
            raise HTTPException(status_code=404, detail="key not found")
        if status == "last_admin":
            # The guard refused — a distinct outcome from an error: the store is
            # fine, the operation was denied to prevent an admin lockout.
            _audit_evt(
                "keys.revoke",
                principal=_p,
                tenant=key_tenant,
                outcome="denied",
                key_id=key_id,
                reason="last_admin",
            )
            raise HTTPException(
                status_code=409,
                detail="refusing to revoke the last admin key (would lock out admin)",
            )
        _audit_evt("keys.revoke", principal=_p, tenant=key_tenant, key_id=key_id)
        return {"revoked": True, "id": key_id}

    @app.get("/api/quotas")
    def list_quotas(_p=_admin) -> dict[str, Any]:
        from mnemostack.quotas import QuotaStoreError

        assert quota_store is not None  # guaranteed by _admin (auth on)
        try:
            return {"quotas": quota_store.list_quotas()}
        except QuotaStoreError as e:
            raise HTTPException(status_code=503, detail=f"quota store unreadable: {e}") from e

    @app.put("/api/quotas/{tenant:path}")
    def set_quota(
        tenant: str,
        body: dict[str, Any] = Body(default_factory=dict),  # noqa: B008 — FastAPI DI
        _p=_admin,
    ) -> dict[str, Any]:
        """Set a tenant's quota (partial update). A field PRESENT in the body is
        applied (``null`` clears it); an ABSENT field is left unchanged."""
        from mnemostack.quotas import _UNSET, QuotaStoreError

        assert quota_store is not None  # guaranteed by _admin (auth on)
        if not tenant.strip():
            raise HTTPException(status_code=400, detail="tenant is required")
        fields = ("max_points", "max_rps", "burst")
        if not any(f in body for f in fields):
            # An empty body would provision an all-unset ({}) record for a new
            # tenant — a no-op PUT shouldn't create one. Require at least one field.
            raise HTTPException(
                status_code=400, detail=f"provide at least one of: {', '.join(fields)}"
            )
        kw = {field: body[field] if field in body else _UNSET for field in fields}
        try:
            quota = quota_store.set(tenant.strip(), **kw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except QuotaStoreError as e:
            _audit_evt(
                "quota.set", principal=_p, tenant=tenant.strip(), outcome="error", error=str(e)
            )
            raise HTTPException(status_code=503, detail=f"quota store error: {e}") from e
        # The RESULTING quota (partial update), so the trail shows what now applies.
        _audit_evt(
            "quota.set",
            principal=_p,
            tenant=tenant.strip(),
            max_points=quota.max_points,
            max_rps=quota.max_rps,
            burst=quota.effective_burst(),
        )
        return {
            "tenant": tenant.strip(),
            "max_points": quota.max_points,
            "max_rps": quota.max_rps,
            "burst": quota.effective_burst(),
        }

    @app.delete("/api/quotas/{tenant:path}")
    def remove_quota(tenant: str, _p=_admin) -> dict[str, Any]:
        from mnemostack.quotas import QuotaStoreError

        assert quota_store is not None  # guaranteed by _admin (auth on)
        try:
            removed = quota_store.remove(tenant)
        except QuotaStoreError as e:
            _audit_evt("quota.remove", principal=_p, tenant=tenant, outcome="error", error=str(e))
            raise HTTPException(status_code=503, detail=f"quota store error: {e}") from e
        if removed:
            _audit_evt("quota.remove", principal=_p, tenant=tenant)
        else:
            _audit_evt(
                "quota.remove", principal=_p, tenant=tenant, outcome="error", reason="not_found"
            )
        return {"removed": bool(removed), "tenant": tenant}

    @app.get("/api/audit")
    def audit_trail(limit: int = Query(200, ge=1, le=1000), _p=_admin) -> dict[str, Any]:
        """The tail of the audit trail (admin-gated). ``enabled=False`` when no
        ``MNEMOSTACK_AUDIT_FILE`` is configured — the UI says how to turn it on.
        Reading the trail is itself a read, not a mutation — not audited."""
        from mnemostack.audit import AuditLogError, FileAuditLog, audit_log_from_env

        sink = audit_log_from_env()
        if not isinstance(sink, FileAuditLog):
            return {"enabled": False, "events": [], "skipped": 0}
        try:
            events, skipped = sink.tail(limit)
        except AuditLogError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return {"enabled": True, "events": events, "skipped": skipped}

    return app


def _row(
    pid: Any,
    payload: dict[str, Any],
    *,
    score: float | None = None,
    text_key: str = "text",
) -> dict[str, Any]:
    p = payload or {}
    row: dict[str, Any] = {
        "id": str(pid),
        # The collection's own text key (a foreign-schema mount) — otherwise
        # the browse table shows every record with an empty text column.
        "text": p.get(text_key, ""),
        "source": p.get("source", ""),
        "invalidated": "invalidated_at" in p,
        "payload": p,
    }
    if score is not None:
        row["score"] = round(score, 4)
    return row
