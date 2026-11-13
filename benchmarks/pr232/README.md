# Retrieval comparison for PR #232

PR #232 changed the order of the ranking stages: `CalibrateRelevance` moved before `ExactTokenRescue`, and `GravityDampen` runs after `FreshnessBlend`. This harness measures whether retrieval quality changed between the commit before the PR and its merge result.

| | Commit |
|---|---|
| Base | `1023cf4c6fa5cee19bc253f711456df9d1b34b8f` |
| Merge result | `d347dddb8e37483012552c0e20c89c4e573e02cc` |

Later changes are in neither tree.

## Result

Run with the embedding model `bge-m3`, 3,000 memories, 240 queries, top 25. The files are in `evidence/`.

- No query ranks worse after the merge, in any configuration.
- 53 of 240 queries rank better with the stateless pipeline, and 54 with the full pipeline and with the serving flow. All of them are exact-code and date queries.
- The order ablation gives the same numbers as the comparison of the two trees. The stage order alone explains the change.

Stateless pipeline:

| Kind | Tree | MRR | hit@3 | hit@5 | recall@10 | recall@25 |
|---|---|---:|---:|---:|---:|---:|
| overall | base | 0.2886 | 0.3667 | 0.4167 | 0.5417 | 0.8042 |
| overall | merge | 0.3166 | 0.4667 | 0.5083 | 0.5458 | 0.8333 |
| exact_code | base | 0.2614 | 0.4000 | 0.4500 | 1.0000 | 1.0000 |
| exact_code | merge | 0.4125 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| date | base | 0.0914 | 0.0000 | 0.1250 | 0.2500 | 0.8250 |
| date | merge | 0.1078 | 0.0000 | 0.1250 | 0.2750 | 1.0000 |

Paired, by the rank of the first gold result:

| Configuration | Better | Worse | Same | Sign test p |
|---|---:|---:|---:|---:|
| raw (no pipeline) | 0 | 0 | 240 | 1 |
| stateless | 53 | 0 | 187 | 2.2e-16 |
| full_fresh_state | 54 | 0 | 186 | 1.1e-16 |
| recall_flow | 54 | 0 | 186 | 1.1e-16 |

The kinds `old_vs_recent`, `paraphrase`, `fallback` and `plain` have the same rank in both trees for every query.

## Limits

- These are retrieval metrics. No answer is generated, so the run says nothing about answer accuracy.
- The corpus is synthetic and adversarial on purpose. Absolute values do not predict quality on real data; only the difference between the two trees is the result. With the pipeline, the absolute metrics are lower than `raw` in both trees on this corpus.
- One embedding model. The LLM reranker, query expansion, the graph arm and the vector floor are off.
- The corpus has no future-dated gold and no malformed payloads, so those fixes of the PR are not measured.

## Method

- Both trees run the same harness. The tree is selected only by `PYTHONPATH`.
- Embeddings go through each tree's own `OllamaProvider`. The base run fills a disk cache. The merge run and the ablation read it with `--cache-readonly`, and a cache miss stops the run. Both trees therefore rank identical vectors.
- The store is an in-memory Qdrant. With `T3_QDRANT_URL` the harness uses a server: it creates a collection with a random name and deletes it at the end.
- Retrieval uses the vector, BM25 and temporal arms with equal-weight RRF and the default candidate budgets.
- The clock of the ranking stages and of the temporal parser is frozen at `2026-05-15T12:00:00+00:00`. The harness checks at start that the freeze works and stops otherwise.
- Configurations: `raw` (recall only), `stateless` (the tree's `build_stateless_pipeline()`), `full_fresh_state` (the tree's `build_full_pipeline` with a new state store per query), `recall_flow` (the composition the HTTP server uses).
- `ablation.py` uses the merge-result tree only. It applies the same stage objects in both orders to the same recalled candidates, so every other fix of the PR stays constant.
- `compare.py` aligns the queries of two result files and reports metric differences, rank wins and losses, top-5 and top-10 crossings, the 30 largest changes in each direction and an exact two-sided sign test.

## Corpus

`corpus.py` writes `corpus.json` with a fixed seed: 3,000 memories and 240 queries, 40 per kind, about one fifth in Russian. No LLM takes part. Every query has one gold memory, which is correct by construction:

| kind | Construction and gold-label justification |
|---|---|
| exact_code | Forty distinct service contexts, cycling through ports, IPs, versions, tickets and error codes. The sole approved active production setting is gold, in MEMORY.md or TOOLS.md. Other notes explicitly describe staging, superseded settings, rejected proposals or an unrelated count using the same token. Distractors span rescue-target and other sources. Ticket/error cases intentionally test the stage's narrower numeric rescue regex too. |
| date | Ten each for ISO dates, spelled-out April 30, 2025, last month and bare 2024. The committee's approved venue is gold; timestamp matches the requested day/month/year. Other decisions have different dates and outcomes. Bare-year gold omits the year in its text, leaving timestamp retrieval available. |
| old_vs_recent | The original signed lease location is explicitly recorded only in gold, 180–687 days old. Four weakly related recent notes concern stationery, catalogs, reminders and labels, aged two minutes, an hour, a day and three days. Two minutes is inside both trees' default ten-minute echo window. |
| paraphrase | Eight synonym/cross-language question-answer templates across five distinct seasonal/visit contexts. Content vocabulary is disjoint across the query and gold, including Russian-to-English and English-to-Russian examples. Function words can overlap. Wrong lexical matches repeat the question but explicitly record no answer or a withdrawn suggestion. Context disambiguates repeated topics. Gold is available to vectors but not lexical matching on content terms. |
| fallback | Recovery-phrase notes have explicit vault identifiers. Gold notes are in Qdrant but omitted from BM25; the sole lexical distractor explicitly lacks the phrase. For these queries only, a VectorRetriever subclass returns no primary hits. The recaller's genuine direct-store fallback uses the same cached query vector and unchanged product merge/scoring logic. Queries contain no temporal expressions, so temporal retrieval contributes nothing. These are controlled failure-path tests, not claims of naturally occurring vector outages. Each row records whether fallback ran and whether gold reached its fallback candidate set. |
| plain | Breakfast tea preferences explicitly stated for each person/context. Coffee after dinner and tea bought as a gift are wrong by construction. Lexical and semantic evidence overlap. |

For the `fallback` kind the harness makes the primary vector arm return nothing for these queries and asserts that the recaller's own fallback ran.

## Run

Make two checkouts of the repository, one at each commit, and point the script at their `src` directories:

```bash
export T3_BASE_SRC=/path/to/base-checkout/src
export T3_MERGED_SRC=/path/to/merged-checkout/src
export T3_OLLAMA_HOST=http://localhost:11434
export T3_EMBED_MODEL=bge-m3
./run.sh
```

Optional: `T3_PYTHON` (the interpreter of an environment with the mnemostack dependencies, default `python`) and `T3_QDRANT_URL`.

The script writes a new `run-XXXXXXXX/` directory with `base.json`, `merged.json`, `comparison.md`, `per_query.csv`, `ablation.md` and the metric tables.

An offline check without an embedding server uses a deterministic fake provider. It tests that the harness runs on both trees, not retrieval quality:

```bash
PYTHONPATH=$T3_BASE_SRC python harness.py --self-test --cache selftest-cache --out selftest-base.json --label BASE
PYTHONPATH=$T3_MERGED_SRC python harness.py --self-test --cache selftest-cache --cache-readonly --out selftest-merged.json --label MERGED
python compare.py selftest-base.json selftest-merged.json
```

## Evidence

- `evidence/per_query.csv`: both ranks, the verdict and the stage flags of the gold result, per query and configuration.
- `evidence/comparison.md`: the full comparison report.
- `evidence/ablation.md`, `evidence/ablation-per-query.csv`: the order ablation.
- `evidence/base-metrics.md`, `evidence/merged-metrics.md`, `evidence/ablation-metrics.md`: the metric tables.

The raw result files with all ranked ids and scores (8 MB per tree) are not in the repository. `run.sh` writes them again.
