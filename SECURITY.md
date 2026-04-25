# Security Policy

## Supported Versions

mnemostack is in alpha (0.2.x). Only the latest alpha release receives
security fixes.

| Version | Supported |
| --- | --- |
| 0.2.0a3+ | ✅ |
| < 0.2.0a3 | ❌ |

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

The HTTP server has no built-in authentication or rate limiting. It binds to
`127.0.0.1` by default; if you expose it with `--host 0.0.0.0`, put it behind
your own auth, TLS, and rate-limit layer.

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
