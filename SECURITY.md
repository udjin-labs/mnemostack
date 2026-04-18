# Security Policy

## Supported Versions

mnemostack is in alpha (0.1.x). Only the latest alpha release receives
security fixes.

| Version | Supported |
| --- | --- |
| 0.1.0a12+ | ✅ |
| < 0.1.0a12 | ❌ |

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
  surfaced in recall results (prompt injection propagation).
- Untrusted queries to `mnemostack serve` endpoints.
- Dependency vulnerabilities in `qdrant-client`, `neo4j`, `fastapi`, etc.
- Secrets leaking into logs or responses.

Out of scope: operator-side misconfiguration (for example exposing the HTTP
server to the internet without an auth proxy) is a deployment concern that
mnemostack does not attempt to fix in-package.

## Dependencies

mnemostack pins upper bounds on its direct dependencies in `pyproject.toml`.
We rely on GitHub's Dependabot and Python's advisory database; if you spot a
transitive vulnerability we missed, please report it through the same private
channel.
