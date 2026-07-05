"""Read-only web inspector — a memory operations console (not a marketing app).

A small, dependency-light, **read-only** operator UI for looking at what's in a
mnemostack deployment: tenants, per-tenant collection/graph size, stored records
(with filters), stale/invalidated facts, dependency reachability, and a quick
smoke query. It never writes — no ingest, no invalidate, no delete — so it's
safe to point at production.

Tenant-aware from day one: every data read is scoped by the selected tenant via
the vector store's tenant boundary, so one tenant's view can't show another's
records. The tenant list is derived from the data (distinct `tenant_id` values)
for now; once service keys land it will come from the key store instead, and the
selected tenant will be resolved from the caller's key rather than chosen freely.

Run it separately from the serving API (it is an operator tool, not the public
recall surface):

    mnemostack inspect --host 127.0.0.1 --port 8100

Install the server extra: ``pip install 'mnemostack[server]'``.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "FastAPI is not installed. Install the optional server extra: "
        "`pip install 'mnemostack[server]'`."
    ) from e

from qdrant_client.models import Filter

from mnemostack import __version__
from mnemostack.config import model_kwargs
from mnemostack.embeddings import get_provider
from mnemostack.server import ServerConfig
from mnemostack.vector import VectorStore
from mnemostack.vector.qdrant import (
    TENANT_ID_KEY,
    _hide_invalidated_condition,
    _tenant_condition,
)

log = logging.getLogger(__name__)

# One self-contained page: no external scripts/styles/fonts (CSP-safe), plain
# and functional. Vanilla JS talks to the read-only /api/* endpoints below.
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
 input[type=text]{min-width:16rem}
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
</style></head><body>
<h1>mnemostack inspector <span class="muted mono" id="ver"></span> <span class="pill">read-only</span></h1>
<div class="bar">
 <label>tenant <select id="tenant"></select></label>
 <input type="text" id="q" placeholder="smoke query (vector search) — empty = browse records">
 <input type="text" id="filters" placeholder='filters JSON e.g. {"source":"notes.md"}' style="min-width:20rem">
 <button id="go">Search</button>
 <span class="muted" id="status"></span>
</div>
<div class="cards" id="cards"></div>
<table><thead><tr><th>id</th><th>source</th><th>text</th><th>state</th></tr></thead>
<tbody id="rows"></tbody></table>
<pre id="detail" hidden></pre>
<script>
const $=s=>document.querySelector(s), esc=t=>String(t??"").replace(/[<&>"']/g,c=>({"<":"&lt;","&":"&amp;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function j(u){const r=await fetch(u);if(!r.ok)throw new Error(r.status+" "+await r.text());return r.json();}
function tenant(){return $("#tenant").value;}
async function loadTenants(){
  const d=await j("/api/tenants");$("#ver").textContent="v"+d.version;
  const sel=$("#tenant");sel.innerHTML="";
  if(!d.tenants.length){sel.innerHTML='<option value="">(no tenants — data has no tenant_id)</option>';}
  for(const t of d.tenants){const o=document.createElement("option");o.value=t.id;o.textContent=t.id+" ("+t.count+")";sel.appendChild(o);}
}
async function loadOverview(){
  const t=tenant();if(t==="")return;
  const d=await j("/api/overview?tenant="+encodeURIComponent(t));
  const dep=(ok,label)=>`<div class="card"><b class="${ok===null?'off':ok?'ok':'bad'}">${ok===null?'—':ok?'up':'down'}</b>${label}</div>`;
  $("#cards").innerHTML=
    `<div class="card"><b>${d.points}</b>points</div>`+
    `<div class="card"><b>${d.invalidated}</b>invalidated</div>`+
    dep(d.qdrant,"qdrant")+dep(d.memgraph,"memgraph (graph)");
}
async function loadRecords(){
  const t=tenant();if(t==="")return;
  $("#status").textContent="loading…";
  const p=new URLSearchParams({tenant:t,limit:"50"});
  if($("#q").value)p.set("q",$("#q").value);
  if($("#filters").value)p.set("filters",$("#filters").value);
  let d;try{d=await j("/api/records?"+p);}catch(e){$("#status").textContent=e.message;return;}
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
$("#go").addEventListener("click",loadRecords);
$("#q").addEventListener("keydown",e=>{if(e.key==="Enter")loadRecords();});
(async()=>{try{await loadTenants();await loadOverview();await loadRecords();}catch(e){$("#status").textContent=e.message;}})();
</script></body></html>"""


def _graph_reachable(cfg: ServerConfig) -> bool | None:
    """Ping the graph if configured. None = not configured (not an error)."""
    if not cfg.graph_uri:
        return None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            cfg.graph_uri,
            auth=(cfg.graph_user, cfg.graph_password) if cfg.graph_user else None,
            connection_timeout=cfg.graph_health_timeout,
            connection_acquisition_timeout=cfg.graph_health_timeout,
        )
        with driver.session(database=cfg.graph_database) as s:
            s.run("RETURN 1").single()
        driver.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def build_inspector_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or ServerConfig.from_env()
    provider = get_provider(cfg.provider_name, **model_kwargs(cfg.embedding_model))
    store = VectorStore(
        collection=cfg.collection, dimension=provider.dimension, host=cfg.qdrant_url
    )

    app = FastAPI(title="mnemostack inspector", version=__version__)

    def _qdrant_ok() -> bool:
        try:
            store.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INSPECTOR_HTML

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/tenants")
    def tenants(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
        """Distinct tenant_id values (from the data) with per-tenant counts.

        Facet-based — cheap and approximate for very large corpora. When service
        keys land, the tenant list moves to the key store.
        """
        out: list[dict[str, Any]] = []
        try:
            resp = store.client.facet(
                collection_name=cfg.collection, key=TENANT_ID_KEY, limit=limit
            )
            out = [{"id": h.value, "count": h.count} for h in resp.hits]
        except Exception as e:  # noqa: BLE001 — facet unsupported / empty collection
            log.info("tenant facet unavailable: %s", e)
        return {"tenants": out, "version": __version__}

    @app.get("/api/overview")
    def overview(tenant: str = Query(..., min_length=1)) -> dict[str, Any]:
        points = 0
        invalidated = 0
        try:
            points = store.count(tenant=tenant)
            invalidated = store.client.count(
                collection_name=cfg.collection,
                # points of this tenant that DO carry an invalidated_at marker
                # (must_not "is-current" == has invalidated_at).
                count_filter=Filter(
                    must=[_tenant_condition(tenant)],
                    must_not=[_hide_invalidated_condition()],
                ),
            ).count
        except Exception as e:  # noqa: BLE001
            log.warning("overview count failed for tenant=%s: %s", tenant, e)
        return {
            "tenant": tenant,
            "collection": cfg.collection,
            "points": points,
            "invalidated": invalidated,
            "qdrant": _qdrant_ok(),
            "memgraph": _graph_reachable(cfg),
            "version": __version__,
        }

    @app.get("/api/records")
    def records(
        tenant: str = Query(..., min_length=1),
        q: str | None = Query(None),
        filters: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        """Records for one tenant. With `q`, a vector smoke search; otherwise a
        browse (scroll). Every read is tenant-scoped — never cross-tenant."""
        import json

        parsed: dict[str, Any] | None = None
        if filters:
            try:
                parsed = json.loads(filters)
                if not isinstance(parsed, dict):
                    return {"records": [], "error": "filters must be a JSON object"}
            except json.JSONDecodeError as e:
                return {"records": [], "error": f"invalid filters JSON: {e}"}

        rows: list[dict[str, Any]] = []
        try:
            if q:
                vec = provider.embed(q)
                for hit in store.search(vec, limit=limit, filters=parsed, tenant=tenant):
                    rows.append(_row(hit.id, hit.payload, score=hit.score))
                mode = "vector search"
            else:
                for hit in store.scroll(filters=parsed, tenant=tenant):
                    rows.append(_row(hit.id, hit.payload))
                    if len(rows) >= limit:
                        break
                mode = "browse"
        except Exception as e:  # noqa: BLE001 — a malformed filter/embed error is a
            # user/runtime problem, not a server bug: return a clean message like
            # the JSON-parse path, never a raw 500 (mirrors /api/overview).
            log.info("records query failed for tenant=%s: %s", tenant, e)
            return {"records": [], "error": f"query failed: {e}", "tenant": tenant}
        return {"records": rows, "mode": mode, "tenant": tenant}

    return app


def _row(pid: Any, payload: dict[str, Any], *, score: float | None = None) -> dict[str, Any]:
    p = payload or {}
    row: dict[str, Any] = {
        "id": str(pid),
        "text": p.get("text", ""),
        "source": p.get("source", ""),
        "invalidated": "invalidated_at" in p,
        "payload": p,
    }
    if score is not None:
        row["score"] = round(score, 4)
    return row
