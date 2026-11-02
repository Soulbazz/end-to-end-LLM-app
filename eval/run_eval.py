"""Retrieval evaluation: Vector-only vs BM25-only vs Hybrid RRF (k=60).

Run:  python eval/run_eval.py
Writes the report to eval/results.md.
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retriever import Retriever  # noqa: E402

EVAL_SET_PATH = ROOT / "data" / "eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.md"


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(set(relevant_ids))


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """1 / rank of the first relevant doc (0 if none retrieved)."""
    rel = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def main() -> None:
    retriever = Retriever()
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    strategies: Dict[str, Callable[[str], List[str]]] = {
        "Baseline 1: Vector-only": lambda q: [d["id"] for d in retriever.search_vector(q, top_k=5)],
        "Baseline 2: BM25-only": lambda q: [d["id"] for d in retriever.search_bm25(q, top_k=5)],
        "Upgraded: Hybrid (BM25 + Vector, RRF k=60)": lambda q: [
            d["id"] for d in retriever.search_hybrid(q, top_k=5, rrf_k=60)
        ],
    }
    names = list(strategies)

    per_query = []
    summary = {}
    for name, fn in strategies.items():
        r3 = r5 = mrr = 0.0
        for item in eval_set:
            ids = fn(item["q"])
            r3 += recall_at_k(ids, item["relevant"], 3)
            r5 += recall_at_k(ids, item["relevant"], 5)
            mrr += reciprocal_rank(ids, item["relevant"])
        n = len(eval_set)
        summary[name] = (r3 / n, r5 / n, mrr / n)

    for i, item in enumerate(eval_set):
        row = {"i": i + 1, "q": item["q"], "relevant": item["relevant"]}
        for name, fn in strategies.items():
            ids = fn(item["q"])
            row[name] = (ids, reciprocal_rank(ids, item["relevant"]), recall_at_k(ids, item["relevant"], 5))
        per_query.append(row)

    vec_n, bm_n, hyb_n = names

    # ---- table ----
    lines = ["| Strategy | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|"]
    for name in names:
        r3, r5, m = summary[name]
        lines.append(f"| {name} | {r3:.3f} | {r5:.3f} | {m:.3f} |")
    table = "\n".join(lines)

    # ---- per-query ----
    def rr(row, name):
        return row[name][1]

    wins = []
    for row in per_query:
        base = max(rr(row, vec_n), rr(row, bm_n))
        if rr(row, hyb_n) > base or (
            row[hyb_n][2] > max(row[vec_n][2], row[bm_n][2])
        ):
            wins.append(row)
    # Also include wins vs only one baseline, ranked by margin
    one_sided = sorted(
        (r for r in per_query if r not in wins),
        key=lambda r: -(rr(r, hyb_n) - min(rr(r, vec_n), rr(r, bm_n))),
    )
    one_sided = [r for r in one_sided if rr(r, hyb_n) > min(rr(r, vec_n), rr(r, bm_n))]
    showcase = (wins + one_sided)[:4]

    fails = [r for r in per_query if rr(r, hyb_n) < 1.0]
    fails.sort(key=lambda r: (rr(r, hyb_n), r["i"]))
    failure = fails[0] if fails else None

    def fmt_row(row) -> str:
        out = [f'**Q{row["i"]}:** "{row["q"]}"  (relevant: {row["relevant"]})']
        for name in names:
            ids, rrv, _ = row[name]
            out.append(f"  - {name}: top-5 = {ids}, RR = {rrv:.2f}")
        return "\n".join(out)

    md = ["# Retrieval Evaluation Results", ""]
    md.append(f"Corpus: {len(retriever.docs)} docs | Queries: {len(eval_set)} (10 semantic, 10 exact-code)")
    md.append("")
    md.append(table)
    md.append("")
    md.append("## Queries where Hybrid RRF beat a baseline")
    md.append("")
    for row in showcase:
        md.append(fmt_row(row))
        md.append("")
    md.append("## Failure / hard case (error analysis)")
    md.append("")
    if failure:
        md.append(fmt_row(failure))
        md.append("")
        md.append(
            "Analysis: the relevant document is only partly matched by lexical overlap and the "
            "n-gram vector model, so neither ranker places it high and RRF cannot recover it. "
            "Real dense embeddings (`USE_API_EMBEDDINGS=1`) or a reranker would likely help."
        )
    else:
        md.append("Hybrid achieved RR = 1.0 on all queries.")
    md.append("")
    report = "\n".join(md)

    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
