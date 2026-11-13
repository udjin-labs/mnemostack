"""Paired retrieval report; exact two-sided binomial sign test, no SciPy."""

import argparse
import csv
import json
import math
from pathlib import Path


def sign_test(better, worse):
    n = better + worse
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2**n)


def report(a, b, csv_path="per_query.csv", config_pairs=None):
    for key in ("corpus_sha256", "model", "self_test", "limit"):
        if a[key] != b[key]:
            raise ValueError(f"incomparable runs: {key}")
    pairs = config_pairs or [(c, c) for c in a["metrics"] if c in b["metrics"]]
    print(
        f"# {a['label']} versus {b['label']}\n\nDifferences are second minus first. Missing rank is worse than any observed rank. MRR uses the evaluated candidate pool; recall_flow ranks are censored at its output limit.\n"
    )
    csv_rows = []
    for ca, cb in pairs:
        ar = {r["query_id"]: r for r in a["rows"] if r["configuration"] == ca and "error" not in r}
        br = {r["query_id"]: r for r in b["rows"] if r["configuration"] == cb and "error" not in r}
        if ar.keys() != br.keys():
            # The metric tables come from each run's own rows. They are
            # comparable only when both runs evaluated the same queries.
            raise ValueError(
                f"incomparable runs: {ca} / {cb} evaluated different queries: "
                f"{sorted(ar.keys() ^ br.keys())[:10]}"
            )
        print(f"## {ca} / {cb}\n")
        print("| kind | metric | first | second | difference |\n|---|---|---:|---:|---:|")
        for kind in sorted(a["metrics"][ca].keys() & b["metrics"][cb].keys()):
            for metric, x in a["metrics"][ca][kind].items():
                y = b["metrics"][cb][kind][metric]
                print(f"| {kind} | {metric} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")
        joined = []
        for qid in sorted(ar.keys() & br.keys()):
            x, y = ar[qid], br[qid]
            rx, ry = x["rank"] or math.inf, y["rank"] or math.inf
            verdict = "better" if ry < rx else "worse" if ry > rx else "same"
            row = dict(
                configuration=ca,
                query_id=qid,
                kind=x["kind"],
                query=x["text"],
                base_rank=x["rank"],
                merged_rank=y["rank"],
                verdict=verdict,
                base_flags=json.dumps(x["gold_flags"], ensure_ascii=False),
                merged_flags=json.dumps(y["gold_flags"], ensure_ascii=False),
            )
            for k in (5, 10):
                row[f"entered_top{k}"] = rx > k and ry <= k
                row[f"left_top{k}"] = rx <= k and ry > k
            csv_rows.append(row)
            # Reciprocal-rank change treats unobserved ranks consistently.
            joined.append(
                ((1 / ry if ry < math.inf else 0) - (1 / rx if rx < math.inf else 0), row)
            )
        print(
            "\n| kind | pairs | better | worse | same | entered 5 | left 5 | entered 10 | left 10 | sign p |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for kind in ["overall"] + sorted({r["kind"] for _, r in joined}):
            rs = [r for _, r in joined if kind == "overall" or r["kind"] == kind]
            n = {v: sum(r["verdict"] == v for r in rs) for v in ("better", "worse", "same")}
            counts = [
                sum(r[f"{verb}_top{k}"] for r in rs)
                for k in (5, 10)
                for verb in ("entered", "left")
            ]
            print(
                f"| {kind} | {len(rs)} | {n['better']} | {n['worse']} | {n['same']} | "
                + " | ".join(map(str, counts))
                + f" | {sign_test(n['better'], n['worse']):.6g} |"
            )
        for title, selected in [
            (
                "regressions",
                sorted((p for p in joined if p[1]["verdict"] == "worse"), key=lambda p: p[0])[:30],
            ),
            (
                "improvements",
                sorted(
                    (p for p in joined if p[1]["verdict"] == "better"),
                    key=lambda p: p[0],
                    reverse=True,
                )[:30],
            ),
        ]:
            print(f"\n### 30 largest {title} (reciprocal-rank change)\n")
            for delta, r in selected:
                print(
                    f"- {r['query_id']} ({r['kind']}): {r['query']} Rank {r['base_rank']} → {r['merged_rank']}; ΔRR {delta:+.4f}. Gold flags: `{r['base_flags']}` → `{r['merged_flags']}`"
                )
        print(
            f"\nUnpaired/error queries: {len(ar.keys() ^ br.keys())}. Recorded skips: first {len(a['skipped'])}, second {len(b['skipped'])}.\n"
        )
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("merged")
    ap.add_argument("--csv", default="per_query.csv")
    args = ap.parse_args()
    report(
        json.loads(Path(args.base).read_text()), json.loads(Path(args.merged).read_text()), args.csv
    )
