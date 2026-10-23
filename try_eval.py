


from datetime import datetime
from pathlib import Path

from src.evaluation.dataset import load_test_cases
from src.evaluation.metrics import mean
from src.evaluation.runner import run_cases

K = 10
cases = [c for c in load_test_cases() if c["category"] == "answerable"]

Path("results").mkdir(exist_ok=True)
out_path = Path("results") / f"eval_k{K}_{datetime.now():%Y%m%d_%H%M%S}.txt"

rows = []
with open(out_path, "w") as f:
    for row in run_cases(cases[:20], k=K, with_judge=True):
        rows.append(row)
        f.write(f"\nQ{row['id']} | {row['question']}\n")
        f.write(f"  relevant:  {row['relevant']}\n")
        f.write(f"  retrieved: {row['retrieved']}\n")
        f.write(f"  recall={row['recall']} ndcg={row['ndcg']} rr={row['rr']}\n")
        f.write(f"\n  ANSWER: {row['answer']}\n")
        f.write(f"  JUDGE: {row['verdict']}\n")

    scored = [r["verdict"] for r in rows if r["verdict"] and "error" not in r["verdict"]]
    failures = [r["id"] for r in rows if r["verdict"] and "error" in r["verdict"]]

    summary = [
        f"k={K}: recall={mean([r['recall'] for r in rows]):.4f}  "
        f"mrr={mean([r['rr'] for r in rows]):.4f}  "
        f"ndcg={mean([r['ndcg'] for r in rows]):.4f}  (n={len(rows)})"
    ]
    for name in ("accuracy", "completeness", "relevance", "groundedness"):
        if scored:
            summary.append(f"judge {name}: {mean([v[name] for v in scored]):.2f}  (n={len(scored)})")
    summary.append(f"judge failures: {len(failures)} {failures}")

    f.write("\n=== SUMMARY ===\n" + "\n".join(summary) + "\n")

print("\n".join(summary))
print(f"details written to {out_path}")