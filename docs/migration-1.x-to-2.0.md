# Migrating from 1.x to 2.0

2.0 makes query/document embedding **role-aware**: known asymmetric model
families automatically get their trained input conventions, collections carry
an embedding-space fingerprint, and mixing incompatible vector spaces is
refused instead of silently degrading recall. For most deployments nothing
changes — but several DEFAULTS changed meaning, which is exactly why this is a
major release. Read the table, find your setup, apply the one relevant step.

## Quick matrix: does anything change for me?

| Your setup | Effect of upgrading | Action |
| --- | --- | --- |
| Gemini, or any model with no built-in profile (e.g. `nomic-embed-text`, MiniLM) | Identity profile — vectors byte-identical to 1.x | None |
| Ollama + `qwen3-embedding:*` | Queries now get the family's instruction prefix; documents unchanged | None — existing collections keep working (query-only transforms never require reindexing) |
| E5 (`e5-*`, `multilingual-e5-*`) via HuggingFace or Ollama | Documents now embed as `passage: …`, queries as `query: …` — a DIFFERENT document space than raw-text 1.x vectors | Reindex into a new collection **or** pin the old behavior (below) |
| HuggingFace + decoder embedding family (`qwen3-embedding`, `e5-mistral`) | Pooling default changed `mean` → `last` (the family's correct mode) | Reindex, or pass `pooling="mean"` explicitly to keep 1.x vectors |
| HuggingFace + explicit `pooling="cls"` (lowercase) | Unchanged | None |
| HuggingFace + `pooling="CLS"` (uppercase) | 1.x silently mean-pooled it; 2.0 normalizes to real CLS pooling — different vectors | Reindex, or pass `pooling="mean"` to keep what 1.x actually did |
| Unknown Ollama model tag | 1.x guessed 768 dimensions; 2.0 probes the live model and FAILS LOUD if unreachable | Ensure the host is reachable at startup (or pass `dimension=` explicitly) |

## Keeping the 1.x behavior without reindexing

Pre-2.0 collections carry no space fingerprint. On upgrade they are treated as
**legacy**: compatible when the active configuration still reproduces raw-text
embedding (identity document transform, legacy pooling), refused loudly when it
does not. Two escape hatches:

- **Profile transforms** — register an identity profile for your model so no
  transform applies (queries and documents stay raw, exactly like 1.x):

  ```python
  from mnemostack.embeddings import EmbeddingProfile, register_embedding_profile

  register_embedding_profile("huggingface", EmbeddingProfile(
      name="raw-legacy", version=1,
      model_patterns=("intfloat/multilingual-e5-large",),  # your exact model
  ))
  ```

- **Pooling** — pass `pooling="mean"` to `HuggingFaceProvider` (constructor or
  your own factory) to keep the 1.x default for decoder families.

The recommended path for asymmetric families is still a clean reindex into a
new collection: the trained conventions measurably improve retrieval.

## Operational changes

- **Space guard**: `index`/`index-markdown`, `Ingestor` and the markdown sync
  refuse to write into a collection whose stamped fingerprint mismatches the
  active configuration — and, on WRITE paths, refuse when the check itself
  cannot be performed (store scroll failure, a write-only credential). Recall
  refuses to query across spaces (`EmbeddingSpaceError`). Verdicts revalidate
  on a bounded interval; `doctor` shows the active profile and fingerprint.
- **Ollama**: embedding uses `POST /api/embed` (batch); the legacy per-item
  endpoint remains only for servers that provably lack it. Default request
  timeout rose 30s → 180s (cold model loads). The host is now honored from
  configuration everywhere: `--ollama-host` > `MNEMOSTACK_OLLAMA_HOST` /
  `embedding.ollama_host` > the native `OLLAMA_HOST` variable > localhost —
  if you previously relied on the config value being ignored, re-check it.
- **`OllamaProvider.embed_batch(..., max_workers=...)`**: the parameter is
  accepted but ignored (the native batch endpoint replaced the thread pool).
- **`ServerConfig` / `mcp.build_server`**: new optional parameters
  (`ollama_host`, `embedding_timeout`) were appended at the positional tail —
  existing positional calls are unaffected.
- **HuggingFace extra**: now requires `transformers>=4.51` (Qwen3
  architectures).

## New knobs (all optional)

`--ollama-host` / `MNEMOSTACK_OLLAMA_HOST` / `embedding.ollama_host`;
`--embedding-timeout` / `MNEMOSTACK_EMBEDDING_TIMEOUT` / `embedding.timeout`;
`--embedding-batch-size` / `MNEMOSTACK_EMBEDDING_BATCH_SIZE` /
`embedding.batch_size`. See `docs/api-stability.md` for their contracts.
