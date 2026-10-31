# One personal archive, two local clients

Scenario: two local agent clients need the same fact from one indexed messenger
archive. This walkthrough uses only a synthetic message and is designed to
exercise shared retrieval. It requires neither shared writes nor a prescribed
personal tenant configuration. Access from two actual PCs is a separate future
verification step with the prerequisites below.

## Local rehearsal

Complete the [quickstart](quickstart.md) through indexing and JSON search. It
creates `archive-demo` with this fictional evidence:

> 2026-10-01 09:15 | Mira | message demo-001
> For the Cedar workshop, the equipment pickup code is CEDAR-482.

The indexed source is `workshop-chat.txt`. Configure two local MCP clients using
the quickstart's configuration. These stand in for agents on two PCs; this
rehearsal does not test a network connection between computers.

In a fresh conversation with each agent, give the same prompt without supplying
the expected code:

> Search the archive for the Cedar workshop equipment pickup code. Report the
> code, the source filename, and the message identifier from the retrieved text.
> If the archive does not contain the answer, say so.

Success means each agent calls `mnemostack_search` and reports `CEDAR-482` with
`workshop-chat.txt` and `demo-001` from the retrieved evidence. Inspect the tool
result as well as the reply, including `degraded` and an optional
`include_trace: true` search trace. Confirm that reranking did not fall back
because the LLM was missing or failed. Do not substitute the retriever label `vector` for
the filename, and do not count an answer without a retrieval call as this test
passing. These are the checks to perform in each client; a tool-only smoke-test does not
verify an agent UI or a connection between physical PCs.

## Future verification on actual PCs A and B

Before repeating the test remotely, the operator must already have:

- An authenticated, protected route from both PCs to the same backend, with
  each agent authorized to read the intended archive. See the
  [deployment guide](../docs/deployment.md) and
  [tenant-auth documentation](../README.md#http-server-optional).
- A client integration configured for that route and the same archive scope.
  The quickstart's stdio configuration connects to local services; copying it
  unchanged onto PC B would point at PC B's localhost, not PC A's archive.
- A backend using the same embedding model for ingestion and retrieval, with
  the synthetic message indexed once. Two independently indexed collections
  would demonstrate separate copies rather than access to one archive.

Repeat the prompt on PC A and PC B and inspect both retrieval results. Record
the backend/collection used and the returned source so the test is reviewable.
This guide does not open ports, expose Qdrant, create credentials, or change
network settings. Remote client configuration depends on your existing setup;
no cross-PC runtime test is claimed here.

## Boundaries and possible follow-up

Supported pieces used here are explicit file indexing, source-bearing search,
and MCP retrieval. Optional answer generation and explicit `mnemostack_remember`
writes already exist, but neither is required for this shared-read example.
There is no automatic messenger export, ongoing capture, or sync configured by
this guide. A possible follow-up is a separate, documented import recipe for a
specific messenger export format, after verifying that format and its source
metadata.

For workplace infrastructure with separate consumers, assign each consumer its
own tenant and enforce authorization at the service boundary. Sharing that
infrastructure does not mean those consumers share the same archive or context.
