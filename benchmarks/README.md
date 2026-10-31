# mnemostack benchmarks

Reproducible LoCoMo benchmark harness for mnemostack. Used to produce the
**82.9%** strict / **92.7%** combined headline numbers on the main README,
plus the earlier `gemini-2.5-flash`-judged 67.8% / 80.4% historical run.

## What gets measured

[LoCoMo](https://github.com/snap-research/locomo) is a long-term dialogue
memory benchmark from Snap Research. Each sample is a multi-session chat
between two speakers with ~500–700 messages; a set of QA probes covers five
categories (single-hop list, temporal, open-domain reasoning, multi-hop,
adversarial). Correct answers must match the gold answer; a judge LLM
decides.

We report:

- **correct** — strict exact-answer match, as judged by the configured LLM (the main README calls this **Strict**)
- **partial** — judge accepts the answer as partially correct
- **wrong** — judge rejects
- **combined** — correct + partial, as a softer aggregate (the main README calls this **Combined**)

## Prerequisites

- Python 3.11+
- A running Qdrant (any port; default `http://localhost:6333`)
- Memgraph is optional; `GraphRetriever` is skipped if unreachable
- A `GEMINI_API_KEY` (used both for embeddings and as the judge LLM; swap to
  Ollama if you prefer local inference — see `--provider` flags)
- About 90 minutes wall-clock for the full 10-sample run with Gemini Flash

## One-shot reproduction

```bash
# from the repo root
pip install -e '.[dev]'
bash benchmarks/download_locomo.sh       # fetches dataset into benchmarks/datasets/
export GEMINI_API_KEY=...
bash benchmarks/run_locomo.sh            # runs all 10 samples, writes results
```

The script's defaults reproduce the published headline configuration:
`window_size=3`, top-K `LIMIT=25`, query expansion on. Override via env
(`LIMIT=15 QUERY_EXPANSION=0 bash benchmarks/run_locomo.sh` for the
conservative config). Results land in `benchmarks/results/<timestamp>.json`
and the full per-QA log in `benchmarks/results/<timestamp>.log`.

## Runners

### Compare two configurations on your own facts

`own_corpus.py` is a trusted-operator tool for an existing Qdrant collection or,
preferably, a separate test copy. It never indexes, deletes, records feedback,
or updates access/IoR state. A tenant selector is not authorization: use credentials
that already permit reading the intended data. Graph recall, query expansion,
foreign payload schemas, and file-backed BM25 are outside this first runner.
Unsupported settings are rejected rather than silently ignored.

Create a JSON list of manually checked queries. Use real stored chunk IDs,
not source filenames. Reserve separate questions for final verification; repeatedly
choosing configurations on these questions does not create an independent test.

```json
[
  {"query_id": "q1", "query": "What is the workshop pickup code?", "expected_ids": ["aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"], "category": "fact"},
  {"query_id": "q2", "query": "An intentionally unsupported question", "expected_ids": [], "category": "negative"}
]
```

Optional per-query fields are `filters`, timezone-aware `as_of`, and
`include_invalidated`. Config files A/B contain supported recall settings,
not credentials or arbitrary application configuration. For example, compare
`{"vector": false, "pipeline": "off", "limit": 10}` with
`{"vector": false, "pipeline": "full", "limit": 10}`:

```bash
python benchmarks/own_corpus.py --queries questions.json \
  --config-a a.json --config-b b.json \
  --qdrant http://127.0.0.1:6333 --collection archive-test \
  --tenant demo --output comparison.json
```

This example uses the existing native BM25 text index without models.
`vector` defaults to true; vector comparisons require explicit
`--allow-model-calls`, `--provider` and, where needed, `--embedding-model` and
`--ollama-host`. Query vectors are prepared once in a batch and shared across A/B.
Their preparation time is separate from retrieval latency, not hidden as free
end-to-end acceleration. `rerank` defaults to false. Enable it only deliberately;
it uses an explicitly configured local Ollama LLM (`--llm-model`), the existing
reranker, and its unchanged 20-candidate window. There is no answer generator or judge.

Both arms use `recall_flow`. Supported settings are `vector`, `text_search`
(`native_bm25`/`off`), `pipeline` (`full`/`off`), `rerank`, `rerank_mode`, `limit`,
`token_budget`, `rrf_k`, `vector_floor`, `hnsw_ef`, `freshness_weight`, and
`access_bonus_max`. Defaults match the corresponding product constructors.
Collection/model/provider are shared controls, not varied silently between arms.

The JSON report includes per-query ranks, paired gains/losses, hit rate,
expected-ID recall, MRR, all-expected coverage, latency with sample counts,
faults versus routine notes, and existing trace/loss reports. Errors contribute
zero to positive-query metrics instead of disappearing. Empty expected-ID lists
form a separate negative category; neither empty results nor score prove absence.
Unknown or foreign expected IDs stay in metric denominators but their identifiers
and contents are not exported. Query/result text is omitted unless `--include-text`.
Provider/exception response bodies and endpoint URLs are not exported.
`ids` and `returned_count` preserve the complete canonical response, including
any extra vector-floor results. `metrics` uses the configured top-K;
`returned_metrics` separately covers the complete returned response.

Each query/arm starts with independent empty in-memory learning state. Ranking
stage time is frozen for the run; relative-date retrieval and external models
remain external inputs. `--cache-mode cold` starts fresh client components;
`warm` runs each arm once before timing it. Both use the same prepared query
vectors. A failed/degraded warm-up is recorded as a failed sample, not timed
as successfully warm. It remains in quality/error denominators; `timed_n`
states the latency sample count. Neither mode flushes Qdrant/OS caches or claims fully cold storage.
A/B order alternates. Tenant-scoped point content/vector hashes are checked
before and after: a changed corpus invalidates the comparison, but matching
hashes do not create a transactional snapshot. Use a stable test copy.
The scan costs work proportional to this tenant's corpus; it is outside query
latency. Exit codes: 0 completed without faults, 1 corpus drift/query faults,
2 setup, input, or report-write failure. Saved artifacts can contain private IDs and must not
be committed or published without an explicit privacy review. Choose a new
output file for each run; the runner refuses to overwrite an existing report.

### Existing benchmark runners

- `locomo_single.py` — single-variant end-to-end run (what the README numbers
  come from). One pipeline configuration, one pass through the dataset.
- `locomo_compare.py` — A/B run comparing raw recall vs full pipeline on the
  same samples. Uses 2x the budget; only needed for tuning work.
- `focused_rerun.py` — replays only the QA that failed in a previous run,
  for fast iteration on prompts and pipeline stages.
- `synthetic_longhorizon.py` — synthetic needle-in-haystack retrieval, no
  LLM judge. Generates dialogue corpora of configurable size (1k → 100k+
  turns) with planted marker facts, then measures `recall@K` and MRR against
  those needles. Useful for seeing where the stack starts to degrade under
  corpus size, complementing LoCoMo's QA-style evaluation.

  ```bash
  # Single scale
  python benchmarks/synthetic_longhorizon.py --turns 10000 --needles 30

  # Multiple scales in one run
  python benchmarks/synthetic_longhorizon.py --scale-steps 5000,10000,25000 \
      --needles 30 --output /tmp/synth.json
  ```

The first three honour `LOCOMO_DATASET` for the dataset path.

## Notes on evaluation protocol

- The same LLM is used for mnemostack's answer generation **and** judging.
  This is the standard LoCoMo setup and matches what Zep / Mem0 / Letta
  publish; a different judge would give different absolute numbers but
  similar relative rankings.
- The current judge is `gemini-3-flash-preview` (since v0.4.0). Earlier
  runs used `gemini-2.5-flash` and are **not directly comparable** on
  absolute scores — the new judge is more lenient on synonyms, partial
  matches, and empty ground truth (that change alone moves v0.3.0 from
  67.8% to 76.7% strict).
- In the current run, the 446 `cat_5` (adversarial open-domain) questions
  have **empty ground truth** and are auto-scored as correct by the
  harness. That inflates the headline aggregate; the main README's
  `signal-only` columns drop those 446 so the score reflects only the
  1540 questions that actually have a ground-truth answer.
- Each `per_qa` row records the recall `degraded` tags for that question.
  Empty-ground-truth rows skip recall entirely, so their `degraded: []`
  means "stack not exercised", not "stack healthy" — only rows that were
  actually answered carry meaningful degradation signals.
- Subset probes (`locomo_single.py --only-questions <file>`) record the
  subset path and selected QA count in the output `config`, so a probe
  artifact is distinguishable from a full run when comparing results.
- We publish the full aggregate across all 5 categories and all 1986 QA.
  Some vendors publish a single best sub-category. See the "Honest numbers
  disclaimer" in the main README.
- The dataset itself lives at
  <https://github.com/snap-research/locomo> under its own license; this
  repository does not redistribute it.

## Synthetic long-horizon results (representative)

Smoke run with 2000 turns / 10 needles on Gemini embeddings + BM25 + RRF
(no pipeline, no LLM):

| Metric | Value |
| --- | --- |
| recall@1 | 0.40 |
| recall@5 | 1.00 |
| recall@20 | 1.00 |
| MRR | 0.70 |
| Query p50 latency | ~300 ms |

What this tells us: pure vector similarity places the correct chunk first
only 40 % of the time because the filler dialogue produces many visually
similar neighbours. RRF with BM25 catches the exact-token (`N-NNNNNNN`)
marker in every case inside the top-5. This is exactly the
"exact-token rescue" pattern we designed for.

Bigger runs (50k+ turns, full pipeline + reranker) are pending; run them
yourself with the command above to see how the curves look on your hardware.
