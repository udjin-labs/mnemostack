# Memory model: corpus, retrieval, and service

mnemostack separates the information you retain, the way you retrieve it, and
who may use it. This guide connects those responsibilities without repeating
the [quickstart](../examples/quickstart.md) or the [deployment reference](deployment.md).

## Choose the corpus and its owner

A corpus contains information your application explicitly ingests. File indexing,
structured markdown, remote writes, and the streaming `Ingestor` are different
ways to populate it. Deterministic chunk IDs and deduplication avoid repeated
writes for unchanged input; they do not decide whether two different sentences
mean the same fact. Enrichment hooks can add fields for filters and answer
context, but the domain-specific extraction function is yours.

Embedding profiles apply the model's query/document conventions. Document-space
fingerprints help ingestion refuse incompatible model or transform changes;
matching vector dimensions alone is insufficient. Keep the write/read embedding
configuration aligned. See [embedding consistency](deployment.md#embedding-model-consistency).

Ownership depends on the use case:

| Arrangement | Corpus and responsibility |
| --- | --- |
| OpenClaw memory | Existing notes can be indexed; the companion plugin decides when to request recall and add retrieved context to the agent prompt. It does not ingest the notes itself. |
| A personal messenger archive used by several agents | One imported archive and one intended owner/scope. Clients need access to that same backend; copying a localhost configuration to another PC does not provide that access. Automatic export, capture, and synchronization need a separate integration. |
| Local memory for a Hermes agent | A local backend serves the agent's memory integration. The separate provider owns its capture/recall wiring; the core does not observe the agent's sessions by itself. |
| Workplace infrastructure | One deployment may serve many consumers, each with its own tenant and keys. Their memories and learning state remain separate; shared infrastructure is not shared context. |

Caller-provided filters select records, but do not authenticate their owner.
For untrusted consumers, enable service-key authentication: the server derives
the tenant and read/write/admin scopes from the key and confines vector, graph,
and learning-state operations accordingly. SDK callers own that trust boundary.
See [authentication](deployment.md#http-server).

## Choose the interface and components

The CLI, HTTP API, and MCP server are ready entry points with their own configuration.
The SDK also exposes components for applications to assemble. They are related
surfaces, not a promise that every bare SDK call enables every feature.

- `Recaller.recall()` collects and fuses configured retrieval sources.
  `recall_flow()` accepts an explicit pipeline and reranker. CLI search/answer
  and the HTTP/MCP serving paths build those components and use that shared flow.
- Vector and temporal retrieval can work without a graph. Lexical sources and
  Memgraph need configuration; a four-source architecture does not mean all four
  are active in the minimal quickstart.
- The built-in serving reranker is LLM-based. A scoring reranker or a custom
  analyzer is an application composition choice; [recipes](recipes.md) show how.
  Query expansion, HyDE, synthesis, and image description have their own opt-in
  or experimental boundaries; see [API stability](api-stability.md).
- `token_budget` cuts the ranked memory-text prefix using the token counter
  (a heuristic by default, replaceable in the SDK). It does not cap the complete
  LLM prompt, generation, or total request cost. CLI tiers instead shape previews.

MCP is useful when an agent speaks that protocol. An application can use HTTP,
CLI, or Python without MCP, and the [OpenClaw companion](../openclaw-plugin/README.md)
can call the HTTP backend. Capture and recall triggers remain integration decisions.

## Maintain facts and their evidence

Suppose a service changes owners. Ingesting a new ownership statement alone does
not retract the old statement. An application can invalidate the superseded
memory, set the world-time interval it knows, and preserve the source of each
statement. Graph facts also support temporal validity.

`invalidated_at` records a system-time retraction; ordinary recall hides stale
facts. `as_of` instead reads `valid_from <= as_of < valid_until`, with missing
bounds open, and ignores `invalidated_at`. It can therefore return facts hidden
from current recall. Without known validity bounds it cannot reconstruct a
history that was never recorded. See [validity contracts](api-stability.md#on-disk--payload-contracts).

A citation points to evidence, but its source may have changed. `resolve` checks
supported local-file citations against the current file and reports whether the
fragment is intact, moved, changed, missing, or unresolvable, or whether only
its surrounding source changed. It is a separate
operation, not an automatic part of recall, and does not certify the statement's
truth. HTTP/MCP resolution additionally requires operator-approved corpus roots.
See [citation audits](recipes.md#auditing-answer-citations-against-their-sources).

Pruning stale chunks, refreshing payloads without re-embedding, explicit
invalidation, hard deletion, and tenant offboarding are shipped lifecycle
operations. They should not be confused with automatic semantic consolidation.
The consolidation `Runtime` executes registered `Phase` objects and reports
results; built-in phases check dependencies, while `CallablePhase` wraps your
logic. Scheduling, contradiction resolution, summarization, promotion, and
forgetting policies require application-supplied work. The separate experimental
`synthesize` API can build an entity profile; it is not a nightly reconciliation
policy. See [consolidation scope](../ARCHITECTURE.md#consolidation--decay).

## Inspect retrieval before tuning it

Use a known source and expected chunk IDs when diagnosing a miss. A trace shows
what each retriever found, the fused list, ranking-stage orders, and reranking.
In Python, `RecallTrace.loss_report(expected_ids)` reports positions and top-k
entries/exits, helping distinguish a candidate that was never retrieved from one
lost later. It diagnoses the recorded path; it does not automatically repair it.

Read `notes` alongside `degraded`: a routine signal such as `temporal:no_parse`
is not a failed component. Preserve response order; `score` is not calibrated
confidence, is not comparable across queries, and need not sort a reranked
response. Answer-model confidence is not a substitute for checking evidence.
See [the score contract](api-stability.md#what-score-is-not).

Explicit usefulness feedback updates learning state. HTTP exposure and access
recording are optional writes. Access reinforcement adds a bounded bonus that
fades toward neutral; it is not a mechanism that punishes or deletes unused
memories. Test tuning on your corpus. The documented LoCoMo runs measure their
specified retrieval/answer setup, not every current runtime feature or deployment.

## Operate a service with explicit limits

Service controls include tenant keys and scopes, quotas, per-tenant metering,
export/offboarding, health and metrics, and an inspector that becomes an admin
console with authentication. An optional OpenBao KV-v2 adapter verifies service
keys; key issuance/revocation belongs to the external store's tools and positive
caching bounds how quickly grant changes take effect.

These are useful operating controls with concrete limits:

- Storage quotas check ingest growth; they are not a globally atomic reservation
  across concurrent workers. HTTP rate limits are per-process. The quota store
  fails open on an unreadable/corrupt file; use additional coordinated controls
  if a hard deployment-wide resource ceiling is required.
- `FileStateStore` uses cross-process `flock` on a local POSIX filesystem. Sharing
  one file across processes on one host is supported; sharing it across hosts or
  over NFS is not a distributed-state solution. Choose a suitable custom store
  or restrict stateful writes for those deployments.
- Tenant export contains vector points and payloads, not a complete service
  backup. Offboarding reports per-store failures and can require a repeat; it is
  not a single transaction spanning every store. Read the
  [offboarding procedure](deployment.md#tenant-offboarding-and-backup).
- The opt-in ingest secret policy blocks selected recognizable token/key formats
  before supported ingestion side effects. It is off by default, is not general
  DLP or prompt-injection protection, and does not scrub old records or inspect
  all later model inputs. See [its exact scope](ingest-policy.md).

Use the [deployment guide](deployment.md) for persistence, auth, backups, metering,
and scaling. Choose the operating model and additional controls that fit your deployment.
