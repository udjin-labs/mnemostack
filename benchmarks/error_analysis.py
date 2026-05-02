#!/usr/bin/env python3
"""Classify LoCoMo wrong answers as retrieval or generation errors.

This script re-runs local retrieval for previously saved benchmark answers. It
uses only the testing Gemini key file by default and a local Qdrant instance.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from mnemostack.embeddings import get_provider
from mnemostack.recall import BM25Doc, build_full_pipeline
from mnemostack.recall.recaller import Recaller
from mnemostack.vector import VectorStore

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results"
TESTING_GEMINI_KEY = Path.home() / ".config" / "gemini" / "api_key.testing"
COLLECTION_PREFIX = "locomo_error_analysis_"

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with",
    "after", "about", "that", "this", "these", "those", "did", "does", "do",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_json", help="Path to LoCoMo results JSON")
    ap.add_argument("--dataset", default=None, help="LoCoMo dataset JSON (auto-detected by default)")
    ap.add_argument("--top-k", type=int, default=15, help="Top-K post-pipeline chunks to inspect")
    ap.add_argument("--raw-limit", type=int, default=50, help="Raw Recaller limit before pipeline")
    ap.add_argument("--max-wrong", type=int, default=None, help="Only analyze first N wrong answers")
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--keep-collections", action="store_true", help="Do not delete collections created by this run")
    ap.add_argument("--output", default=None, help="Detailed JSON output path")
    return ap.parse_args()


def ensure_testing_gemini_key() -> None:
    """Force the benchmark to use the non-production Gemini key."""
    try:
        key = TESTING_GEMINI_KEY.read_text().strip()
    except FileNotFoundError as exc:
        raise SystemExit(f"Testing Gemini key file is missing: {TESTING_GEMINI_KEY}") from exc
    if not key:
        raise SystemExit(f"Testing Gemini key file is empty: {TESTING_GEMINI_KEY}")
    os.environ["GEMINI_API_KEY"] = key


def load_results(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        rows = data.get("per_qa") or data.get("results") or data.get("items")
    else:
        rows = data
    if not isinstance(rows, list):
        raise SystemExit(f"Could not find result rows in {path}")
    return rows


def wrong_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r.get("correct") and not r.get("partial")]


def find_dataset(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_dataset = os.environ.get("LOCOMO_DATASET")
    if env_dataset:
        candidates.append(Path(env_dataset))
    candidates.extend([
        ROOT / "locomo_data" / "locomo10.json",
        ROOT / "locomo_data" / "data" / "locomo10.json",
        ROOT / "datasets" / "locomo10.json",
    ])
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("LoCoMo dataset not found. Tried locomo_data/ and datasets/locomo10.json")


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"Expected dataset list in {path}")
    samples = {str(s.get("sample_id")): s for s in data if s.get("sample_id")}
    if not samples:
        raise SystemExit(f"No sample_id entries found in {path}")
    return samples


def parse_session_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime(2023, 1, 1, tzinfo=timezone.utc)


def sample_messages(sample: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    conv = sample["conversation"]
    sess_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda x: int(x.split("_")[1]),
    )
    texts: list[str] = []
    records: list[dict[str, Any]] = []
    for skey in sess_keys:
        if not isinstance(conv.get(skey), list):
            continue
        sess_date = parse_session_date(conv.get(f"{skey}_date_time", ""))
        for msg in conv[skey]:
            text = f"[{sess_date.strftime('%Y-%m-%d')}] {msg['speaker']}: {msg['text']}"
            texts.append(text)
            records.append({
                "source": f"{sample['sample_id']}/{skey}",
                "sample_id": sample["sample_id"],
                "session": skey,
                "timestamp": sess_date.isoformat(),
                "speaker": msg["speaker"],
                "dia_id": msg.get("dia_id", ""),
            })
    return texts, records


def collection_name(sample_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", sample_id)
    return f"{COLLECTION_PREFIX}{safe}"


def make_store(client: QdrantClient, provider: Any, collection: str) -> VectorStore:
    store = VectorStore.__new__(VectorStore)
    store.collection = collection
    store.dimension = provider.dimension
    store.distance = Distance.COSINE
    store.client = client
    return store


def collection_count(client: QdrantClient, collection: str) -> int | None:
    try:
        info = client.get_collection(collection)
    except Exception:  # noqa: BLE001 - Qdrant versions throw different exceptions for missing collections
        return None
    return int(info.points_count or 0)


def index_sample(
    sample: dict[str, Any],
    provider: Any,
    client: QdrantClient,
    created_collections: set[str],
) -> tuple[VectorStore, list[BM25Doc]]:
    texts, records = sample_messages(sample)
    collection = collection_name(sample["sample_id"])
    existing_count = collection_count(client, collection)
    if existing_count != len(texts):
        if existing_count is not None:
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=provider.dimension, distance=Distance.COSINE),
        )
        vectors = provider.embed_batch(texts)
        points = [
            (i, vec, {"text": text, **rec})
            for i, (text, vec, rec) in enumerate(zip(texts, vectors, records, strict=True), start=1)
            if vec
        ]
        store = make_store(client, provider, collection)
        store.upsert_batch(points, batch_size=100)
        try:
            store.index_payload_field("timestamp", PayloadSchemaType.DATETIME)
        except Exception:  # noqa: BLE001
            pass
        created_collections.add(collection)
    else:
        store = make_store(client, provider, collection)

    bm25 = [
        BM25Doc(id=i, text=text, payload={"text": text, **rec})
        for i, (text, rec) in enumerate(zip(texts, records, strict=True), start=1)
    ]
    return store, bm25


def normalize_dates(text: str) -> str:
    # ISO-ish dates: 2023-05-07 / 2023/05/07 -> add stable date tokens.
    def iso_repl(match: re.Match[str]) -> str:
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return f" {y:04d}-{m:02d}-{d:02d} {y:04d} {m:d} {d:d} "

    text = re.sub(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", iso_repl, text)

    # Day Month Year: 7 May 2023 -> same stable date tokens.
    month_pat = "|".join(sorted(MONTHS, key=len, reverse=True))
    weekday_pat = "|".join(WEEKDAYS)

    def date_tokens(y: int, m: int, d: int) -> str:
        return f" {y:04d}-{m:02d}-{d:02d} {y:04d} {m:d} {d:d} "

    def dmy_repl(match: re.Match[str]) -> str:
        d = int(match.group(1))
        m = MONTHS[match.group(2).lower()]
        y = int(match.group(3))
        return date_tokens(y, m, d)

    # Relative weekday answers such as "Sunday before 25 May 2023" carry a
    # computed answer date, not just the anchor date.
    def relative_weekday_repl(match: re.Match[str]) -> str:
        weekday = match.group(1).lower()
        direction = match.group(2).lower()
        d = int(match.group(3))
        m = MONTHS[match.group(4).lower()]
        y = int(match.group(5))
        anchor = datetime(y, m, d)
        target_weekday = WEEKDAYS[weekday]
        if direction == "before":
            delta = (anchor.weekday() - target_weekday) % 7 or 7
            target = anchor - timedelta(days=delta)
        else:
            delta = (target_weekday - anchor.weekday()) % 7 or 7
            target = anchor + timedelta(days=delta)
        return f" {weekday} {date_tokens(target.year, target.month, target.day)} "

    text = re.sub(
        rf"\b(?:the\s+)?({weekday_pat})\s+(before|after)\s+(\d{{1,2}})\s+({month_pat})\s*,?\s*(\d{{4}})\b",
        relative_weekday_repl,
        text,
        flags=re.I,
    )
    text = re.sub(rf"\b(\d{{1,2}})\s+({month_pat})\s*,?\s*(\d{{4}})\b", dmy_repl, text, flags=re.I)

    # Month Year: June 2023 -> 2023-06 token; preserve year/month numbers.
    def my_repl(match: re.Match[str]) -> str:
        m = MONTHS[match.group(1).lower()]
        y = int(match.group(2))
        return f" {y:04d}-{m:02d} {y:04d} {m:d} "

    text = re.sub(rf"\b({month_pat})\s+(\d{{4}})\b", my_repl, text, flags=re.I)
    return text


def normalize_text(text: Any) -> str:
    value = normalize_dates(str(text).lower())
    table = str.maketrans({ch: " " for ch in string.punctuation if ch not in "-"})
    value = value.translate(table)
    return re.sub(r"\s+", " ", value).strip()


def content_tokens(text: Any) -> list[str]:
    tokens = normalize_text(text).split()
    return [t for t in tokens if t not in STOPWORDS]


def number_tokens(text: Any) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", normalize_text(text)))


def classify_match(ground_truth: Any, chunks: list[str]) -> tuple[str, float, dict[str, Any]]:
    gt_tokens = content_tokens(ground_truth)
    gt_token_set = set(gt_tokens)
    gt_numbers = number_tokens(ground_truth)
    if not gt_token_set:
        return "A", 0.0, {"reason": "empty_ground_truth"}

    best_ratio = 0.0
    best_chunk = ""
    best_missing_numbers: list[str] = []
    keyword_hits = 0

    for chunk in chunks:
        chunk_tokens = set(content_tokens(chunk))
        overlap = gt_token_set & chunk_tokens
        missing_numbers = sorted(gt_numbers - number_tokens(chunk))
        ratio = len(overlap) / max(len(gt_token_set), 1)
        if missing_numbers:
            # Numeric claims are only full matches when all GT numbers appear exactly.
            ratio = min(ratio, 0.59)
        if ratio > best_ratio:
            best_ratio = ratio
            best_chunk = chunk
            best_missing_numbers = missing_numbers
        keyword_hits = max(keyword_hits, len(overlap))

    detail = {
        "token_overlap_ratio": round(best_ratio, 3),
        "gt_tokens": sorted(gt_token_set),
        "gt_numbers": sorted(gt_numbers),
        "missing_numbers": best_missing_numbers,
        "best_chunk": best_chunk,
    }
    if best_ratio > 0.60:
        return "B", best_ratio, detail
    if best_ratio >= 0.25 or keyword_hits >= 2:
        return "C", best_ratio, detail
    return "A", best_ratio, detail


def label_for(code: str) -> str:
    return {"A": "Retrieval miss", "B": "Generation fail", "C": "Partial retrieval"}[code]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    ensure_testing_gemini_key()
    rows = load_results(Path(args.results_json))
    wrong = wrong_only(rows)
    if args.max_wrong is not None:
        wrong = wrong[: args.max_wrong]

    dataset_path = find_dataset(args.dataset)
    samples = load_dataset(dataset_path)
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    provider = get_provider("gemini")
    pipeline = build_full_pipeline()

    cache: dict[str, tuple[VectorStore, list[BM25Doc]]] = {}
    created_collections: set[str] = set()
    details: list[dict[str, Any]] = []
    summary: dict[str, Counter[str]] = defaultdict(Counter)

    try:
        for idx, row in enumerate(wrong, start=1):
            sample_id = str(row.get("sample"))
            if sample_id not in samples:
                raise SystemExit(f"Sample {sample_id!r} not found in {dataset_path}")
            if sample_id not in cache:
                print(f"Indexing {sample_id}...", flush=True)
                cache[sample_id] = index_sample(samples[sample_id], provider, client, created_collections)
            store, bm25 = cache[sample_id]
            recaller = Recaller(embedding_provider=provider, vector_store=store, bm25_docs=bm25)
            raw = recaller.recall(str(row.get("question", "")), limit=args.raw_limit)
            retrieved = pipeline.apply(str(row.get("question", "")), raw)[: args.top_k]
            chunks = [r.text for r in retrieved]
            code, ratio, match_detail = classify_match(row.get("ground_truth", ""), chunks)
            cat = str(row.get("category", "unknown"))
            summary[cat][code] += 1
            result = {
                **row,
                "error_class": code,
                "error_label": label_for(code),
                "match": match_detail,
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "id": r.id,
                        "score": r.score,
                        "sources": r.sources,
                        "text": r.text,
                        "payload": r.payload,
                    }
                    for rank, r in enumerate(retrieved, start=1)
                ],
            }
            details.append(result)
            print(
                f"{idx:>4}/{len(wrong)} {sample_id} cat{cat}: {code} {label_for(code)} "
                f"overlap={ratio:.2f} | {str(row.get('question', ''))[:80]}",
                flush=True,
            )
    finally:
        if not args.keep_collections:
            for collection in sorted(created_collections):
                if collection.startswith(COLLECTION_PREFIX):
                    try:
                        client.delete_collection(collection)
                    except Exception as exc:  # noqa: BLE001
                        print(f"warning: failed to delete {collection}: {exc}", file=sys.stderr)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_results": str(Path(args.results_json)),
        "dataset": str(dataset_path),
        "top_k": args.top_k,
        "raw_limit": args.raw_limit,
        "total_wrong_analyzed": len(details),
        "summary": {cat: dict(counts) for cat, counts in sorted(summary.items())},
        "details": details,
    }


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    print("\nSummary by category")
    print("category  A_retrieval_miss  B_generation_fail  C_partial_retrieval  total")
    for cat, counts in sorted(summary.items(), key=lambda kv: kv[0]):
        a = counts.get("A", 0)
        b = counts.get("B", 0)
        c = counts.get("C", 0)
        print(f"{cat:>8}  {a:>16}  {b:>17}  {c:>19}  {a + b + c:>5}")


def main() -> None:
    args = parse_args()
    result = analyze(args)
    print_summary(result["summary"])
    if args.output:
        output = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = DEFAULT_RESULTS_DIR / f"error_analysis_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
