"""Single-variant LoCoMo runner — tests ONE pipeline end-to-end.

Unlike locomo_compare.py which runs raw + full_pipeline side by side (2x cost),
this runs only the target configuration (full mnemostack with pipeline + graph).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from mnemostack.embeddings import get_provider
from mnemostack.llm import get_llm
from mnemostack.recall import AnswerGenerator, BM25Doc, Recaller, build_full_pipeline
from mnemostack.vector import VectorStore

DATASET = "./datasets/locomo10.json"


def parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime(2023, 1, 1, tzinfo=timezone.utc)


def ingest_sample(sample, provider, client, collection, log):
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
    texts, records = [], []
    for skey in sess_keys:
        if not isinstance(conv.get(skey), list):
            continue
        sess_date = parse_date(conv.get(f"{skey}_date_time", ""))
        for msg in conv[skey]:
            text = f"[{sess_date.strftime('%Y-%m-%d')}] {msg['speaker']}: {msg['text']}"
            texts.append(text)
            records.append({
                "source": f"{sample['sample_id']}/{skey}",
                "session": skey,
                "timestamp": sess_date.isoformat(),
                "speaker": msg["speaker"],
                "dia_id": msg.get("dia_id", ""),
            })

    log(f"  embedding {len(texts)} messages (batch)...")
    t0 = time.time()
    vectors = provider.embed_batch(texts)
    log(f"  embedded in {time.time() - t0:.1f}s")

    points, bm25 = [], []
    for i, (text, vec, rec) in enumerate(zip(texts, vectors, records), start=1):
        if not vec:
            continue
        points.append((i, vec, {"text": text, **rec}))
        bm25.append(BM25Doc(id=i, text=text, payload={"text": text, **rec}))

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
    return store, bm25


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
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--qa", type=int, default=None)
    ap.add_argument("--limit", type=int, default=15, help="top-K memories for answer generator")
    ap.add_argument("--output", default="/tmp/locomo_single.json")
    ap.add_argument("--log", default="/tmp/locomo_single.log")
    args = ap.parse_args()

    with open(DATASET) as f:
        dataset = json.load(f)[: args.samples]

    client = QdrantClient(host="localhost", port=6333)
    provider = get_provider("gemini")
    llm = get_llm("gemini")
    answer_gen = AnswerGenerator(llm=llm, confidence_threshold=0.5, max_memories=args.limit)
    pipeline = build_full_pipeline()  # no graph — dataset ingest makes its own isolated collection

    if os.path.exists(args.log):
        os.remove(args.log)

    def log(msg):
        print(msg, flush=True)
        with open(args.log, "a") as f:
            f.write(msg + "\n")

    stats = {"total": 0, "correct": 0, "partial": 0, "wrong": 0, "by_cat": {}}
    all_per_qa = []

    for si, sample in enumerate(dataset):
        sid = sample["sample_id"]
        collection = f"locomo_single_{sid}"
        log(f"\n=== Sample {si+1}/{len(dataset)}: {sid} ===")
        store, bm25 = ingest_sample(sample, provider, client, collection, log)
        recaller = Recaller(embedding_provider=provider, vector_store=store, bm25_docs=bm25)

        qa_list = sample["qa"] if args.qa is None else sample["qa"][: args.qa]
        for qi, qa in enumerate(qa_list):
            q = qa["question"]
            truth = str(qa.get("answer", ""))
            cat = qa.get("category", 0)

            raw = recaller.recall(q, limit=50)
            mems = pipeline.apply(q, raw)[: args.limit]
            ans = answer_gen.generate(q, mems)
            r = evaluate(q, ans.text, truth, llm)

            stats["total"] += 1
            key = f"cat_{cat}"
            stats["by_cat"].setdefault(key, {"correct": 0, "partial": 0, "wrong": 0})
            if r["correct"]:
                stats["correct"] += 1
                stats["by_cat"][key]["correct"] += 1
                mark = "✓"
            elif r["partial"]:
                stats["partial"] += 1
                stats["by_cat"][key]["partial"] += 1
                mark = "~"
            else:
                stats["wrong"] += 1
                stats["by_cat"][key]["wrong"] += 1
                mark = "✗"
            log(f"  {mark} Q{qi+1} cat{cat}: {q[:60]} | true:{truth[:40]} | pred:{ans.text[:40]}")
            all_per_qa.append({
                "sample": sid, "question": q, "ground_truth": truth,
                "predicted": ans.text, "category": cat, **r,
            })

        try:
            client.delete_collection(collection)
        except Exception:
            pass

    log("\n=== FINAL ===")
    n = stats["total"]
    log(f"Correct: {stats['correct']}/{n} ({100*stats['correct']/max(n,1):.1f}%)")
    log(f"Partial: {stats['partial']}/{n} ({100*stats['partial']/max(n,1):.1f}%)")
    log(f"Wrong:   {stats['wrong']}/{n} ({100*stats['wrong']/max(n,1):.1f}%)")
    log(f"Combined (correct+partial): {100*(stats['correct']+stats['partial'])/max(n,1):.1f}%")
    log("By category:")
    for k, v in sorted(stats["by_cat"].items()):
        t = v["correct"] + v["partial"] + v["wrong"]
        log(f"  {k}: correct={v['correct']}/{t} partial={v['partial']} wrong={v['wrong']}")

    Path(args.output).write_text(json.dumps({"stats": stats, "per_qa": all_per_qa}, ensure_ascii=False, indent=2))
    log(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
