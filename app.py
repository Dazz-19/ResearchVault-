import gradio as gr
import pandas as pd

from src.evaluation.dataset import load_test_cases
from src.evaluation.metrics import mean
from src.evaluation.runner import run_cases

K = 10
SCORE_NAMES = ("accuracy", "completeness", "groundedness", "relevance")

# loaded once when the app starts
cases = [c for c in load_test_cases() if c["category"] == "answerable"]


def card(label, value):
    return f"""
    <div style="margin:8px 0;padding:12px 16px;border-radius:8px;border-left:5px solid #888;background:#8881;">
        <div style="font-size:13px;opacity:.7">{label}</div>
        <div style="font-size:26px;font-weight:bold">{value}</div>
    </div>
    """


def run_retrieval_eval(progress=gr.Progress()):
    """Retrieval only: fast and free. No generation, no judge."""
    rows = []
    total = len(cases)
    for i, row in enumerate(run_cases(cases, k=K, with_judge=False), start=1):
        rows.append(row)
        progress(i / total, desc=f"Retrieval {i}/{total}")

    html = (
        card(f"Recall@{K}", f"{mean([r['recall'] for r in rows]):.4f}")
        + card("MRR", f"{mean([r['rr'] for r in rows]):.4f}")
        + card(f"nDCG@{K}", f"{mean([r['ndcg'] for r in rows]):.4f}")
        + card("Questions", len(rows))
    )

    df = pd.DataFrame(
        [{"id": r["id"], "question": r["question"], "recall": r["recall"],
          "rr": r["rr"], "ndcg": r["ndcg"]} for r in rows]
    ).sort_values("recall", ascending=True)  # failures at the top
    return html, df


def run_judge_eval(progress=gr.Progress()):
    """Generation + LLM judge: slow and costs money."""
    rows = []
    total = len(cases)
    for i, row in enumerate(run_cases(cases, k=K, with_judge=True), start=1):
        rows.append(row)
        progress(i / total, desc=f"Judging {i}/{total}")

    scored = [r["verdict"] for r in rows if r["verdict"] and "error" not in r["verdict"]]
    failed = [r["id"] for r in rows if r["verdict"] and "error" in r["verdict"]]

    averages = {n: mean([v[n] for v in scored]) for n in SCORE_NAMES} if scored else {}

    html = "".join(card(n.capitalize(), f"{averages[n]:.2f} / 5") for n in averages)
    html += card("Judge failures", f"{len(failed)} {failed if failed else ''}")
    html += card("Scored questions", len(scored))

    chart_df = pd.DataFrame(
        {"Metric": [n.capitalize() for n in averages], "Score": list(averages.values())}
    )

    table = []
    for r in rows:
        v = r["verdict"] or {}
        if "error" in v:
            table.append({"id": r["id"], "question": r["question"],
                          **{n: None for n in SCORE_NAMES},
                          "answer": r["answer"], "feedback": v["error"]})
        else:
            table.append({"id": r["id"], "question": r["question"],
                          **{n: v[n] for n in SCORE_NAMES},
                          "answer": r["answer"], "feedback": v["feedback"]})
    table_df = pd.DataFrame(table).sort_values(
        "completeness", ascending=True, na_position="first"  # failures, then weakest answers
    )
    return html, chart_df, table_df


with gr.Blocks(title="ResearchVault Evaluation") as demo:
    gr.Markdown("# ResearchVault Evaluation Dashboard")

    # ---------- Retrieval ----------
    gr.Markdown("## Retrieval evaluation")
    retrieval_btn = gr.Button("Run retrieval evaluation", variant="primary")
    retrieval_cards = gr.HTML("<p style='opacity:.6'>Click the button to start.</p>")
    retrieval_table = gr.Dataframe(label="Per-question retrieval (worst first)", wrap=True)

    # ---------- Judge ----------
    gr.Markdown("## Answer evaluation (LLM as judge)")
    gr.Markdown("Runs generation and judging for every question. This costs API calls.")
    judge_btn = gr.Button("Run answer evaluation", variant="primary")
    with gr.Row():
        with gr.Column(scale=1):
            judge_cards = gr.HTML("<p style='opacity:.6'>Click the button to start.</p>")
        with gr.Column(scale=1):
            judge_chart = gr.BarPlot(x="Metric", y="Score", y_lim=[1, 5],
                                     title="Average judge scores", height=350)
    judge_table = gr.Dataframe(label="Per-question answers and feedback (weakest first)", wrap=True)

    retrieval_btn.click(fn=run_retrieval_eval, outputs=[retrieval_cards, retrieval_table])
    judge_btn.click(fn=run_judge_eval, outputs=[judge_cards, judge_chart, judge_table])


if __name__ == "__main__":
    demo.launch()