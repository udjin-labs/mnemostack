#!/bin/bash
# Reproducible LoCoMo run — the single-variant harness that produced the
# numbers in the project README.
#
# Usage:
#   export GEMINI_API_KEY=...                   # required
#   bash benchmarks/run_locomo.sh               # full 10 samples
#   SAMPLES=3 bash benchmarks/run_locomo.sh     # quick run, 3 samples
#
# Output lands in benchmarks/results/<UTC timestamp>.{json,log}.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$HERE/results"
mkdir -p "$RESULTS_DIR"

SAMPLES="${SAMPLES:-10}"
LIMIT="${LIMIT:-15}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JSON_OUT="$RESULTS_DIR/$STAMP.json"
LOG_OUT="$RESULTS_DIR/$STAMP.log"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "error: GEMINI_API_KEY is not set. Export it before running this script." >&2
    exit 1
fi

DATASET_DEFAULT="$HERE/datasets/locomo10.json"
if [[ -z "${LOCOMO_DATASET:-}" ]] && [[ ! -f "$DATASET_DEFAULT" ]]; then
    echo "LoCoMo dataset not found. Run: bash benchmarks/download_locomo.sh" >&2
    exit 1
fi
export LOCOMO_DATASET="${LOCOMO_DATASET:-$DATASET_DEFAULT}"

echo "Running LoCoMo: samples=$SAMPLES limit=$LIMIT"
echo "  dataset: $LOCOMO_DATASET"
echo "  json:    $JSON_OUT"
echo "  log:     $LOG_OUT"
echo

python3 "$HERE/locomo_single.py" \
    --samples "$SAMPLES" \
    --limit "$LIMIT" \
    --output "$JSON_OUT" \
    --log "$LOG_OUT"

echo
echo "Done."
echo "  JSON:    $JSON_OUT"
echo "  Log:     $LOG_OUT"
echo
echo "Summary (last block of the log):"
tail -n 15 "$LOG_OUT"
