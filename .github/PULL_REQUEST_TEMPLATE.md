## What this changes

Brief description — what problem does this PR solve, and for whom?

## Why

Link to the related issue or design discussion if any (`Closes #123`,
`Refs #456`). If there is no issue, a short rationale is enough.

## How it was tested

- [ ] Unit tests added / updated (`pytest`)
- [ ] Ran the HTTP server / CLI / benchmark against a real service (not only mocks)
- [ ] Docs updated (README / ARCHITECTURE / CHANGELOG)

Please paste the relevant test output or curl/CLI session. Unit tests alone
are not enough for anything that talks to Qdrant, Memgraph, or an LLM.

## Checklist

- [ ] No hard-coded API keys, personal paths, or workspace-specific names
- [ ] Backward-compatible, or breaking change called out explicitly
- [ ] Follows the architecture in `ARCHITECTURE.md` (retrievers, pipeline, etc.)
- [ ] Alpha API: if public API changes, bump `CHANGELOG.md` and add a migration note
