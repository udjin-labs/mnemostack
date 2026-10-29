# Ingest secret policy

The optional `block` policy rejects recognizable secret formats before built-in
ingestion sends the input to an embedding provider or enrichment hook, or stores it.
It is **off by default**. It does not require labels from importers.

## Enable

Set `MNEMOSTACK_SECRET_POLICY=block`, or use the configuration file:

```yaml
ingest:
  secret_policy: "block"
```

The CLI flag `--secret-policy block` overrides configuration for `index`,
`index-markdown`, `serve`, and `mcp-serve`. An HTTP/MCP client cannot disable the
server's policy through request metadata. The policy applies to every tenant;
authentication still determines tenant ownership and write permission.

SDK callers opt in explicitly:

```python
from mnemostack.ingest import IngestItem, Ingestor
from mnemostack.ingest_policy import IngestPolicyError

ingestor = Ingestor(embedding, store, secret_policy="block")
try:
    ingestor.ingest_one(IngestItem(text=document, source="notes"))
except IngestPolicyError:
    # Record a refusal, not the document or matched value.
    pass
```

`MarkdownSyncer`, `collect_markdown`, `upsert_markdown_chunks`,
`ingest_remote_items`, `expand_remote_items`, and `TripleExtractor` also accept
`secret_policy="block"`. Check the complete input before caller-controlled
chunking: a pattern split across separate incoming records cannot be reconstructed
by this policy. Use `expand_remote_items(..., secret_policy="block")` when using
the SDK's remote chunk helper.

## Scope

Checks cover text, source names, tags, and nested metadata, including dictionary
keys. Built-in file ingestion checks complete file text before chunking.
Enrichment output is checked again before storage. HTTP and MCP graph writes
check their structured input; `TripleExtractor` checks text before its LLM call.

Recognized formats are GitHub classic and fine-grained tokens, GitLab PATs with
the standard prefix, Slack bot tokens, Vault `hvs` tokens, and complete PEM/OpenSSH
private-key blocks with supported labels. Private-key labels include PKCS8,
encrypted PKCS8, RSA, EC, and OpenSSH. Traditional encrypted PEM headers are supported.
These are format checks, not proof that a credential works. Rules follow selected
[Gitleaks formats](https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml);
this feature is not the complete Gitleaks scanner.

The policy does not detect all API keys, arbitrary passwords, custom token
prefixes, encoded or incomplete secrets. It does not contact issuers to validate
tokens. It does not inspect search queries or previously stored memories sent
to answer models. Direct low-level store/provider calls bypass ingestion checks.
It is not protection against malicious instructions inside documents.

## Refusals and examples

Matching input raises `IngestPolicyError` in the SDK. HTTP returns 422 without
the original text. MCP returns `ok: false` with `error_kind: "ingest_policy"`.
Malformed HTTP ingestion requests also omit input echoes when blocking is enabled.

HTTP checks the entire memory-write request before processing its first item.
SDK streams can retain safe batches completed before a later refusal. A refused
file is not treated as empty: its existing records are not pruned. Blocking does
not remove any previously stored secret.

Full dummy tokens in documentation can match the same formats as real tokens.
Words such as `test` or `example` do not exempt a match. Use visibly incomplete
placeholders such as `${TOKEN}` in material that must pass this strict policy.
Names, telephone numbers, and email addresses alone are not blocked.
