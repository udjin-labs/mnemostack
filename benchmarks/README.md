# mnemostack benchmarks

Reproducible LoCoMo benchmark harness for mnemostack. Used to produce the
66.4% / 79.2% numbers on the main README.

## What gets measured

[LoCoMo](https://github.com/snap-research/locomo) is a long-term dialogue
memory benchmark from Snap Research. Each sample is a multi-session chat
between two speakers with ~500–700 messages; a set of QA probes covers five
categories (single-hop list, temporal, open-domain reasoning, multi-hop,
adversarial). Correct answers must match the gold answer; a judge LLM
decides.

We report:

- **correct** — strict exact-answer match, as judged by the configured LLM
- **partial** — judge accepts the answer as partially correct
- **wrong** — judge rejects
- **combined** — correct + partial, as a softer aggregate

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

Results land in `benchmarks/results/<timestamp>.json` and the full per-QA
log in `benchmarks/results/<timestamp>.log`.

## Runners

- `locomo_single.py` — single-variant end-to-end run (what the README numbers
  come from). One pipeline configuration, one pass through the dataset.
- `locomo_compare.py` — A/B run comparing raw recall vs full pipeline on the
  same samples. Uses 2x the budget; only needed for tuning work.
- `focused_rerun.py` — replays only the QA that failed in a previous run,
  for fast iteration on prompts and pipeline stages.

All three honour `LOCOMO_DATASET` for the dataset path.

## Notes on evaluation protocol

- The same LLM is used for mnemostack's answer generation **and** judging.
  This is the standard LoCoMo setup and matches what Zep / Mem0 / Letta
  publish; a different judge would give different absolute numbers but
  similar relative rankings.
- We publish the full aggregate across all 5 categories and all 1986 QA.
  Some vendors publish a single best sub-category. See the "Honest numbers
  disclaimer" in the main README.
- The dataset itself lives at
  <https://github.com/snap-research/locomo> under its own license; this
  repository does not redistribute it.
