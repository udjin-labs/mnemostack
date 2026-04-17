#!/usr/bin/env python3
"""LoCoMo benchmark: mnemostack full pipeline vs baseline recall-only.

Reproduces the methodology used in workspace/scripts for LoCoMo, but through
mnemostack API. Compares two configurations:
    A) Recaller raw (BM25 + vector + RRF only)
    B) Recaller + build_full_pipeline (8-stage rerank)

Expects LoCoMo dataset at the path below. Prints combined + correct rates
per variant and saves detailed results to results_*.json.

Usage:
    GEMINI_API_KEY=... python3 benchmarks/locomo_compare.py --samples 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from mnemostack.chunking import MessagePairChunker
from mnemostack.embeddings import get_provider
from mnemostack.llm import get_llm
from mnemostack.recall import (
    AnswerGenerator,
    BM25Doc,
    Recaller,
    build_full_pipeline,
)
from mnemostack.vector import VectorStore

DATASET = Path(
    "./datasets/locomo10.json"
)


def parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime(2023, 1, 1, tzinfo=timezone.utc)


def ingest_sample(sample, provider, client, collection, log, pair_chunks=False):
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=provider.dimension, distance=Distance.COSINE),
    )

    conv = sample["conversation"]
    sess_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda x: int(x.split("_")[1]),
    )

    raw_texts, raw_records = [], []
    for skey in sess_keys:
        if not isinstance(conv.get(skey), list):
            continue
        date_key = f"{skey}_date_time"
        sess_date = parse_date(conv.get(date_key, ""))
        for msg in conv[skey]:
            raw_texts.append(f"[{sess_date.strftime('%Y-%m-%d')}] {msg['speaker']}: {msg['text']}")
            raw_records.append({
                "source": f"{sample['sample_id']}/{skey}",
                "session": skey,
                "timestamp": sess_date.isoformat(),
                "speaker": msg["speaker"],
                "dia_id": msg.get("dia_id", ""),
            })
    if pair_chunks:
        chunker = MessagePairChunker(include_solo=True, window=2)
        chunks = chunker.chunk_messages(raw_texts, metadata=raw_records)
        texts = [c.text for c in chunks]
        records = [c.metadata for c in chunks]
    else:
        texts, records = raw_texts, raw_records

    log(f"  embedding {len(texts)} messages (batch)...")
    t0 = time.time()
    vectors = provider.embed_batch(texts)
    log(f"  embedded in {time.time() - t0:.1f}s")

    points = []
    bm25_docs = []
    for i, (text, vec, rec) in enumerate(zip(texts, vectors, records), start=1):
        if not vec:
            continue
        points.append((i, vec, {"text": text, **rec}))
        bm25_docs.append(BM25Doc(id=i, text=text, payload={"text": text, **rec}))

    store = VectorStore.__new__(VectorStore)
    store.collection = collection
    store.dimension = provider.dimension
    store.distance = Distance.COSINE
    store.client = client
    store.upsert_batch(points, batch_size=100)
    try:
        store.index_payload_field("timestamp", PayloadSchemaType.DATETIME)
    except Exception:
        pass
    return store, bm25_docs


def run_variant(
    *,
    name,
    recaller,
    pipeline,
    answer_gen,
    qa_list,
    limit,
    log,
):
    """Run all QA with given config, return stats + per-q results."""
    stats = {"total": 0, "correct": 0, "partial": 0, "wrong": 0, "by_cat": {}}
    per_qa = []
    for qi, qa in enumerate(qa_list):
        q = qa["question"]
        truth = str(qa.get("answer", ""))
        cat = qa.get("category", 0)

        raw = recaller.recall(q, limit=50)
        memories = pipeline.apply(q, raw)[:limit] if pipeline else raw[:limit]
        ans = answer_gen.generate(q, memories)
        eval_r = evaluate(q, ans.text, truth, answer_gen.llm)

        stats["total"] += 1
        cat_key = f"cat_{cat}"
        stats["by_cat"].setdefault(cat_key, {"correct": 0, "partial": 0, "wrong": 0})
        if eval_r["correct"]:
            stats["correct"] += 1
            stats["by_cat"][cat_key]["correct"] += 1
        elif eval_r["partial"]:
            stats["partial"] += 1
            stats["by_cat"][cat_key]["partial"] += 1
        else:
            stats["wrong"] += 1
            stats["by_cat"][cat_key]["wrong"] += 1

        mark = "✓" if eval_r["correct"] else ("~" if eval_r["partial"] else "✗")
        log(f"  [{name}] {mark} Q{qi+1} cat{cat}: {q[:50]} | true:{truth[:40]} | pred:{ans.text[:40]}")

        per_qa.append({
            "question": q,
            "ground_truth": truth,
            "predicted": ans.text,
            "confidence": ans.confidence,
            "category": cat,
            **eval_r,
            "variant": name,
        })
    return stats, per_qa


def evaluate(q, pred, truth, llm):
    prompt = (
        f"Evaluate factual answer. Query: {q}\nGround truth: {truth}\n"
        f"Predicted: {pred}\n\nRespond JSON only: "
        '{"correct": true|false, "partial": true|false, "reason": "..."}'
    )
    resp = llm.generate(prompt, max_tokens=100)
    r = resp.text.strip()
    if r.startswith("```"):
        r = r.split("\n", 1)[1] if "\n" in r else r[3:]
    if r.endswith("```"):
        r = r[:-3]
    try:
        return json.loads(r)
    except Exception:
        return {"correct": truth.lower().strip() in pred.lower(), "partial": False, "reason": "fallback"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--qa", type=int, default=None)
    ap.add_argument("--pair-chunks", action="store_true", help="Use MessagePairChunker for sliding-window chunks")
    ap.add_argument("--output", default="./benchmarks/locomo_compare_results.json")
    args = ap.parse_args()

    with open(DATASET) as f:
        dataset = json.load(f)[: args.samples]

    client = QdrantClient(host="localhost", port=6333)
    provider = get_provider("gemini")
    llm = get_llm("gemini")
    answer_gen = AnswerGenerator(llm=llm, confidence_threshold=0.5)
    pipeline = build_full_pipeline()

    LOG_FILE = "/tmp/locomo_compare.log"
    def log(msg):
        print(msg, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")

    log(f"=== LoCoMo compare: mnemostack raw vs full-pipeline @ {datetime.now().isoformat()} ===")

    all_a_stats = {"total": 0, "correct": 0, "partial": 0, "wrong": 0}
    all_b_stats = {"total": 0, "correct": 0, "partial": 0, "wrong": 0}
    by_cat_a, by_cat_b = {}, {}
    per_qa_all = []

    for sidx, sample in enumerate(dataset):
        sid = sample["sample_id"]
        collection = f"locomo_cmp_{sid}"
        log(f"\n=== Sample {sidx+1}/{len(dataset)}: {sid} ===")
        store, bm25_docs = ingest_sample(sample, provider, client, collection, log, pair_chunks=args.pair_chunks)
        recaller = Recaller(embedding_provider=provider, vector_store=store, bm25_docs=bm25_docs)

        qa = sample["qa"]
        if args.qa:
            qa = qa[:args.qa]

        log(f"--- variant A: Recaller raw top-K ---")
        a_stats, a_qa = run_variant(
            name="raw",
            recaller=recaller,
            pipeline=None,
            answer_gen=answer_gen,
            qa_list=qa,
            limit=15,
            log=log,
        )
        log(f"--- variant B: Recaller + full pipeline ---")
        b_stats, b_qa = run_variant(
            name="full_pipeline",
            recaller=recaller,
            pipeline=pipeline,
            answer_gen=answer_gen,
            qa_list=qa,
            limit=15,
            log=log,
        )

        for target, src in [(all_a_stats, a_stats), (all_b_stats, b_stats)]:
            for k in ("total", "correct", "partial", "wrong"):
                target[k] += src[k]
        for target, src in [(by_cat_a, a_stats["by_cat"]), (by_cat_b, b_stats["by_cat"])]:
            for k, v in src.items():
                target.setdefault(k, {"correct": 0, "partial": 0, "wrong": 0})
                for s in ("correct", "partial", "wrong"):
                    target[k][s] += v[s]

        per_qa_all.extend(a_qa)
        per_qa_all.extend(b_qa)

        try:
            client.delete_collection(collection)
        except Exception:
            pass

    def pct(s):
        if s["total"] == 0:
            return "(no data)"
        return (
            f"correct={s['correct']} ({s['correct']/s['total']*100:.1f}%) "
            f"partial={s['partial']} ({s['partial']/s['total']*100:.1f}%) "
            f"wrong={s['wrong']}"
        )

    log("\n=== FINAL ===")
    log(f"Variant A (raw):           {pct(all_a_stats)}")
    log(f"Variant B (full-pipeline): {pct(all_b_stats)}")
    log(f"\nBy category — raw:")
    for k, v in sorted(by_cat_a.items()):
        t = v["correct"] + v["partial"] + v["wrong"]
        log(f"  {k}: correct={v['correct']}/{t} partial={v['partial']}")
    log(f"\nBy category — full pipeline:")
    for k, v in sorted(by_cat_b.items()):
        t = v["correct"] + v["partial"] + v["wrong"]
        log(f"  {k}: correct={v['correct']}/{t} partial={v['partial']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "raw": {"summary": all_a_stats, "by_category": by_cat_a},
            "full_pipeline": {"summary": all_b_stats, "by_category": by_cat_b},
            "per_qa": per_qa_all,
        }, f, ensure_ascii=False, indent=2)
    log(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
