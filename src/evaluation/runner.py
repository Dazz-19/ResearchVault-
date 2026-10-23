from src.evaluation.converters import convert_retrieved_results, convert_test_relevant_questions
from src.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from src.retrieval import retrieve_with_rerank
from src.generation import generate_from_results
from src.evaluation.judge import judge_answer


def run_cases(cases, k=10, with_judge=False):
    """Yield one result dict per case. Prints nothing and writes no files."""
    for case in cases:
        relevant = convert_test_relevant_questions(case["relevant_pages"])
        raw_retrieved = retrieve_with_rerank(case["question"], k=k)
        retrieved = convert_retrieved_results(raw_retrieved)

        row = {
            "id": case["id"],
            "question": case["question"],
            "category": case["category"],
            "relevant": relevant,
            "retrieved": retrieved,
            "recall": recall_at_k(retrieved, relevant, k=k),
            "ndcg": ndcg_at_k(retrieved, relevant, k=k),
            "rr": reciprocal_rank(retrieved, relevant, k=k),
            "answer": None,
            "verdict": None,
        }

        if with_judge:
            try:
                row["answer"] = generate_from_results(case["question"], raw_retrieved)["answer"]
                row["verdict"] = judge_answer(case, row["answer"], raw_retrieved)
            except Exception as e:
                row["verdict"] = {"error": f"{type(e).__name__}: {e}"}

        yield row