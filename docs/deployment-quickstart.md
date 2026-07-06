# Deployment quickstart

The 5-minute path from nothing to a running mnemostack that answers recall
queries. For the full production reference (systemd units, Docker Compose,
security, backups, monitoring) see [deployment.md](deployment.md).

> Conventions: `mnemostack` is the CLI (`pip install mnemostack`); the HTTP
> server needs the extra `pip install 'mnemostack[server]'`. All commands read
> config from a YAML file / `MNEMOSTACK_*` env vars — see
> [config.md](config.md). Nothing here needs a cloud account.

---

## 1. Minimal local deployment

You need **Qdrant** (required) and an **embedding provider** (Gemini via
`GEMINI_API_KEY`, or a local Ollama). Memgraph is optional.

```bash
# 1. Qdrant (Docker; or run the binary)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# 2. Install mnemostack + the HTTP server extra
pip install 'mnemostack[server]'

# 3. Point it at your provider (Gemini shown; or use --provider ollama)
export GEMINI_API_KEY=...

# 4. Verify the deployment BEFORE indexing — read-only, safe to run anywhere
mnemostack doctor
```

`doctor` exits `0` when config + embedding + Qdrant are healthy, `1` if a core
dependency is down, `2` if the config is invalid — so it drops straight into a
deploy gate. A fresh Qdrant with no data reports the collection as "does not
exist yet" (a warning, not a failure).

```bash
# 5. Index some data, then query it
mnemostack index ./notes            # or: index-markdown ./vault
mnemostack search "your question"
mnemostack answer "your question"   # needs an LLM configured
```

## 2. Run the HTTP server

```bash
mnemostack serve --host 127.0.0.1 --port 8000 --memgraph-uri ""
```

Applications in any language then call `/recall`, `/answer`, `/feedback`. See
the [HTTP API section of the README](../README.md#http-api) for request shapes.

> Binding to `0.0.0.0` exposes an **unauthenticated** API — only do it behind
> your own auth / rate-limit layer or a private network.

> `--memgraph-uri ""` keeps graph recall **off** in this minimal setup. Without
> it, `serve` falls back to the default `bolt://localhost:7687` and every recall
> pays a Bolt connection timeout until Memgraph is running — enable it
> deliberately in step 3.

## 3. Add Memgraph (optional — graph recall)

Graph memory is optional and fail-soft: recall works without it. To enable it:

```bash
docker run -d --name memgraph -p 7687:7687 memgraph/memgraph

# Enable graph on the server + index links (markdown [[wikilinks]] etc.)
export MNEMOSTACK_MEMGRAPH_URI=bolt://localhost:7687   # auth: MNEMOSTACK_GRAPH_USER/_PASSWORD
mnemostack index-markdown ./vault --memgraph-uri "$MNEMOSTACK_MEMGRAPH_URI"
mnemostack serve --memgraph-uri "$MNEMOSTACK_MEMGRAPH_URI"
```

> The `serve` CLI defaults `--memgraph-uri` to `bolt://localhost:7687`. To run
> **without** a graph, pass `--memgraph-uri ""` explicitly (omitting the flag
> does not disable it — the server would log repeated Bolt timeouts).

## 4. On-prem deployment path

Everything above runs air-gapped — mnemostack has no hosted dependency of its
own. For a fully local stack:

- **Embeddings**: use `--provider ollama` (a local Ollama server) instead of a
  hosted API — no key, no egress.
- **Qdrant / Memgraph**: self-hosted (Docker, binary, or systemd) on the same
  host or private network; point `MNEMOSTACK_QDRANT_URL` /
  `MNEMOSTACK_MEMGRAPH_URI` at them.
- **Config**: ship a `mnemostack.yaml` (see `mnemostack init`) alongside the
  service rather than env vars; `mnemostack config` prints the effective config.
- Run `mnemostack doctor` as a post-deploy gate on every host.

## 5. Health & readiness verification

Wire these into your process manager / orchestrator:

| Check | What it tells you | Use for |
| --- | --- | --- |
| `GET /healthz` | process is up (no backend checks) | Kubernetes `livenessProbe` |
| `GET /readyz` | Qdrant + embedding reachable (graph is fail-soft, not gated) | Kubernetes `readinessProbe` / LB |
| `GET /status` | version, config, live dependency reachability, degradation counters | operator dashboard |
| `GET /metrics` | Prometheus counters + latency histograms | scraping |
| `mnemostack doctor` | full read-only diagnostic + remediation hints | deploy/CI gate, on-call triage |

```bash
curl -fsS http://localhost:8000/healthz          # {"status":"ok",...}
curl -fsS http://localhost:8000/readyz           # 503 until Qdrant+embedding are ready
curl -fsS http://localhost:8000/metrics | head
mnemostack doctor --json                         # machine-readable full report
```

## 6. Smoke test

On an installed host, verify the deployment with the built-in diagnostics — no
source tree required:

```bash
mnemostack health     # component reachability: embedding + Qdrant
mnemostack doctor     # deeper checks: config, dimension match, LLM, and graph
```

From a **source checkout**, a runnable smoke set (in-memory Qdrant + fake
embedder — no external services) exercises ingest → recall, the stale-fact
validity view, and markdown collection. Target the file directly so pytest
doesn't import the rest of the test tree (which needs the full `.[dev]` extra):

```bash
pytest tests/test_smoke.py -q
```

To smoke a **live** Memgraph too, set `MNEMOSTACK_SMOKE_GRAPH_URI` before
running (the graph smoke is skipped otherwise).

---

Next: [deployment.md](deployment.md) for production hardening, and
[api-stability.md](api-stability.md) for which APIs are stable vs experimental.
