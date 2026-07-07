# Security Policy

## Supported Versions

mnemostack is stable and actively developed. Only the latest minor release
receives security fixes.

| Version | Supported |
| --- | --- |
| latest minor release | ✅ |
| older                | ❌ |

> Note: 0.5.0 and earlier contain a known tenant-isolation issue in filtered
> recall (several retrievers ignored `filters=`). If you rely on payload
> filters for isolation, upgrade to the first release containing the fix —
> see the Security section of `CHANGELOG.md`.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Instead, report it privately through GitHub's private vulnerability reporting:

1. Go to <https://github.com/udjin-labs/mnemostack/security/advisories/new>
2. Fill in the advisory draft with:
   - A clear description of the issue and its impact
   - Step-by-step reproduction instructions
   - Any proof-of-concept code if applicable
   - Suggested fix if you have one

You should get an acknowledgement within 3 working days. After triage we'll
agree on a disclosure timeline — typically 30 days for medium-severity and
14 days for high-severity issues, unless the fix requires longer.

## Threat model in scope

mnemostack is a library and a small HTTP service consumers run themselves.
Relevant attack surfaces:

- Untrusted memory contents injected into the vector store or graph and then
  surfaced in recall results or LLM prompts (prompt injection propagation).
- Untrusted queries to `mnemostack serve` endpoints.
- Dependency vulnerabilities in `qdrant-client`, `neo4j`, `fastapi`, etc.
- Secrets leaking into logs, responses, prompts, or third-party provider traffic.

The HTTP server is **unauthenticated by default** (single-tenant): it binds to
`127.0.0.1`, and if you expose it with `--host 0.0.0.0` you must put it behind
your own TLS, auth, and rate-limit layer. For multi-tenant deployments it also
has an opt-in, built-in **service-key auth** mode: `mnemostack serve --auth`
requires a per-tenant service key on every data endpoint (default-deny — `401`
without a valid key, `403` for an insufficient scope), resolves the tenant from
the key rather than the request, and enforces that boundary across the vector
store, the tenant-scoped knowledge graph, and per-tenant learning state. Each
tenant can be capped with `mnemostack quota set` — storage (`--max-points`,
enforced at ingest) and request rate (`--max-rps`, enforced on the authenticated
HTTP surface, returning `429`). The MCP server has an equivalent process-bound
`mcp-serve --auth`. Caveats: the operator endpoints (`/health`, `/healthz`,
`/readyz`, `/status`, `/metrics`) stay unauthenticated **even under `--auth`** and
expose version/reachability/counters — protect them at your proxy if sensitive;
and the rate limiter is **per-process** (N workers multiply the effective ceiling
by N), so use a reverse proxy for a hard global limit. Auth is off by default —
single-tenant deployments are unaffected.

Recall `filters=` provide data isolation inside the retrievers, but they are
caller-supplied — **not an authorization boundary**: a client that can reach the
endpoint can pass any filter, or none. For a real trust boundary in multi-tenant
deployments, use `--auth` (above) instead of trusting client filters; a
proxy/gateway-injected filter is still useful as defense-in-depth.

LLM-backed features (`/answer`, LLM reranking, HyDE, triple extraction, query
expansion) send retrieved memory text and/or user queries to the configured LLM
provider. Use a local provider such as Ollama, redact sensitive payloads, or
disable those features when memories contain data that must not leave your
environment. Treat retrieved memory text as untrusted prompt input.

## Dependencies

mnemostack pins upper bounds on its direct dependencies in `pyproject.toml`.
We rely on GitHub's Dependabot and Python's advisory database; if you spot a
transitive vulnerability we missed, please report it through the same private
channel.
