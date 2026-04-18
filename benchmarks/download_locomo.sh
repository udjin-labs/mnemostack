#!/bin/bash
# Download the LoCoMo10 dataset into benchmarks/datasets/locomo10.json.
#
# Source: https://github.com/snap-research/locomo (raw file under data/).
# This script just fetches one JSON file; it does not redistribute the
# dataset inside this repo.
#
# Usage: bash benchmarks/download_locomo.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$HERE/datasets"
DEST="$DEST_DIR/locomo10.json"
# Official upstream location. Update here if Snap reorganises the repo.
URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

mkdir -p "$DEST_DIR"

if [[ -s "$DEST" ]]; then
    echo "LoCoMo dataset already present at $DEST — skipping download."
    exit 0
fi

echo "Downloading LoCoMo10 dataset from $URL ..."
if command -v curl >/dev/null 2>&1; then
    curl --fail --location --output "$DEST" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$DEST" "$URL"
else
    echo "error: need curl or wget on PATH" >&2
    exit 1
fi

echo "OK. Dataset saved to $DEST ($(du -h "$DEST" | cut -f1))."
