#!/usr/bin/env python3
"""Re-run only the failing QA from a previous LoCoMo compare_results.json.

Much faster iteration loop for prompt tuning. Reuses ingestion per sample
but only evaluates QA that previously failed (wrong or partial).
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from mnemostack.chunking import MessagePairChunker
from mnemostack.embeddings import get_provider
from mnemostack.llm import get_llm
from mnemostack.recall import AnswerGenerator, BM25Doc, QueryExpander, Recaller, build_full_pipeline
from mnemostack.vector import VectorStore

DATASET = Path('./datasets/locomo10.json')


def parse_date(s):
    try:
        return datetime.strptime(s, '%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime(2023, 1, 1, tzinfo=timezone.utc)


def ingest(sample, provider, client, collection, pair_chunks=True):
    try: client.delete_collection(collection)
    except Exception: pass
    client.create_collection(collection, vectors_config=VectorParams(size=provider.dimension, distance=Distance.COSINE))
    conv = sample['conversation']
    sess_keys = sorted([k for k in conv if k.startswith('session_') and not k.endswith('date_time')], key=lambda x: int(x.split('_')[1]))
    msgs, metas = [], []
    for skey in sess_keys:
        if not isinstance(conv.get(skey), list): continue
        sess_date = parse_date(conv.get(f'{skey}_date_time', ''))
        for msg in conv[skey]:
            msgs.append(f"[{sess_date.strftime('%Y-%m-%d')}] {msg['speaker']}: {msg['text']}")
            metas.append({'source': f"{sample['sample_id']}/{skey}", 'timestamp': sess_date.isoformat()})
    if pair_chunks:
        chunker = MessagePairChunker(include_solo=True, window=2)
        chunks = chunker.chunk_messages(msgs, metadata=metas)
        texts = [c.text for c in chunks]
        records = [c.metadata for c in chunks]
    else:
        texts, records = msgs, metas
    vecs = provider.embed_batch(texts)
    points = [(i, v, {'text': t, **r}) for i,(t,v,r) in enumerate(zip(texts,vecs,records),1) if v]
    bm25 = [BM25Doc(id=i, text=t, payload={'text': t, **r}) for i,(t,v,r) in enumerate(zip(texts,vecs,records),1) if v]
    store = VectorStore.__new__(VectorStore)
    store.collection = collection
    store.dimension = provider.dimension
    store.distance = Distance.COSINE
    store.client = client
    store.upsert_batch(points, batch_size=100)
    return store, bm25


def evaluate(q, pred, truth, llm):
    p = f"Evaluate factual answer. Query: {q}\nGround truth: {truth}\nPredicted: {pred}\n\nRespond JSON only: " + '{"correct": true|false, "partial": true|false, "reason": "..."}'
    r = llm.generate(p, max_tokens=100).text.strip()
    if r.startswith('```'): r = r.split('\n',1)[1] if '\n' in r else r[3:]
    if r.endswith('```'): r = r[:-3]
    try: return json.loads(r)
    except Exception: return {'correct': truth.lower().strip() in pred.lower(), 'partial': False, 'reason': 'fallback'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prev', required=True, help='path to previous compare_results.json')
    ap.add_argument('--variant', default='full_pipeline', choices=['raw','full_pipeline'])
    ap.add_argument('--include', default='wrong,partial', help='comma list: wrong,partial,correct')
    ap.add_argument('--output', default='/tmp/focused_rerun.json')
    args = ap.parse_args()

    include = set(args.include.split(','))
    with open(args.prev) as f:
        prev = json.load(f)
    failing = []
    for r in prev.get('per_qa', []):
        if r['variant'] != args.variant: continue
        label = 'correct' if r['correct'] else ('partial' if r['partial'] else 'wrong')
        if label in include:
            failing.append(r)
    # group by sample_id from per_qa? we need sample context
    with open(DATASET) as f:
        dataset = json.load(f)
    by_sid = {s['sample_id']: s for s in dataset}
    # map q -> sid by scanning dataset
    q_to_sid = {}
    for s in dataset:
        for qa in s['qa']:
            q_to_sid[qa['question']] = s['sample_id']

    # group
    by_sample = defaultdict(list)
    for r in failing:
        sid = q_to_sid.get(r['question'])
        if sid: by_sample[sid].append(r)

    print(f'Total failing to rerun: {len(failing)} across {len(by_sample)} samples', flush=True)

    client = QdrantClient(host='localhost', port=6333)
    provider = get_provider('gemini')
    llm = get_llm('gemini')
    ag = AnswerGenerator(llm=llm, confidence_threshold=0.5, max_memories=20)
    use_expansion = True
    pipeline = build_full_pipeline() if args.variant == 'full_pipeline' else None

    transitions = {'fixed': 0, 'still_wrong': 0, 'still_partial': 0, 'regressed': 0}
    per_qa = []

    for sid, rs in by_sample.items():
        sample = by_sid[sid]
        print(f'\n=== {sid}: ingesting ({len(sample["conversation"])} sessions) ===', flush=True)
        store, bm25 = ingest(sample, provider, client, f'focused_{sid}')
        rec = Recaller(embedding_provider=provider, vector_store=store, bm25_docs=bm25)
        qe = QueryExpander(recaller=rec, llm=llm, n_variants=3) if use_expansion else None
        for r in rs:
            q = r['question']; truth = str(r['ground_truth']); old = 'wrong' if not r['correct'] and not r['partial'] else 'partial'
            raw = (qe.recall(q, limit=50) if qe else rec.recall(q, limit=50))
            mems = pipeline.apply(q, raw)[:20] if pipeline else raw[:20]
            ans = ag.generate(q, mems)
            ev = evaluate(q, ans.text, truth, llm)
            new = 'correct' if ev['correct'] else ('partial' if ev['partial'] else 'wrong')
            status = {'wrong':{'correct':'fixed','partial':'fixed','wrong':'still_wrong'},
                      'partial':{'correct':'fixed','partial':'still_partial','wrong':'regressed'}}[old][new]
            transitions[status] += 1
            mark = {'correct':'✓','partial':'~','wrong':'✗'}[new]
            tag = {'fixed':'  ★FIXED','still_wrong':'  ·','still_partial':'  ·','regressed':'  ↓REGRESS'}[status]
            print(f"  {mark} cat{r['category']} old={old} new={new}{tag}")
            print(f"     Q: {q[:80]}")
            print(f"     T: {truth[:80]}")
            print(f"     P: {ans.text[:80]}")
            per_qa.append({'question': q, 'category': r['category'], 'old': old, 'new': new, 'status': status, 'ground_truth': truth, 'predicted': ans.text, 'reason': ev.get('reason','')})
        try: client.delete_collection(f'focused_{sid}')
        except: pass

    print(f'\n=== SUMMARY ===')
    for k, v in transitions.items(): print(f'  {k}: {v}')
    score = transitions['fixed'] - transitions['regressed']
    print(f'  NET: {score:+d} (fixed - regressed)')
    with open(args.output, 'w') as f:
        json.dump({'transitions': transitions, 'per_qa': per_qa}, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
