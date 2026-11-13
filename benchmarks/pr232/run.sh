#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${T3_OLLAMA_HOST:?Set T3_OLLAMA_HOST}"
: "${T3_EMBED_MODEL:?Set T3_EMBED_MODEL}"
: "${T3_BASE_SRC:?Set T3_BASE_SRC to the src directory of a checkout at the base commit}"
: "${T3_MERGED_SRC:?Set T3_MERGED_SRC to the src directory of a checkout at the merge commit}"
PY=${T3_PYTHON:-python}
BASE=$T3_BASE_SRC
MERGED=$T3_MERGED_SRC
extra=()
if [[ -n ${T3_QDRANT_URL:-} ]]; then extra+=(--qdrant-url "$T3_QDRANT_URL"); fi
# A fresh run directory prevents accidentally mixing caches from different weights.
run_dir=$(mktemp -d ./run-XXXXXXXX)
timeout 300 "$PY" corpus.py
common=(--corpus corpus.json --ollama-host "$T3_OLLAMA_HOST" --embedding-model "$T3_EMBED_MODEL" --cache "$run_dir/cache" --limit 25 "${extra[@]}")
# Real embedding runs can take hours; timeout covers offline validation only.
PYTHONPATH="$BASE" "$PY" harness.py "${common[@]}" --out "$run_dir/base.json" --label BASE > "$run_dir/base-metrics.md"
PYTHONPATH="$MERGED" "$PY" harness.py "${common[@]}" --cache-readonly --out "$run_dir/merged.json" --label MERGED > "$run_dir/merged-metrics.md"
"$PY" compare.py "$run_dir/base.json" "$run_dir/merged.json" --csv "$run_dir/per_query.csv" > "$run_dir/comparison.md"
PYTHONPATH="$MERGED" "$PY" ablation.py "${common[@]}" --cache-readonly --out "$run_dir/ablation.json" --label MERGED-order-ablation --report "$run_dir/ablation.md" --csv "$run_dir/ablation-per-query.csv" > "$run_dir/ablation-metrics.md"
printf 'Results: %s\n' "$run_dir"
