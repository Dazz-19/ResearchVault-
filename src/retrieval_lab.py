import pandas as pd
from src.vectorstore import load_vectorstore

K = 5


def run_retrieval(csv_path="data/evaluations.csv", k=K):
    """Yield (row, results) for one question at a time."""
    df = pd.read_csv(csv_path)
    store = load_vectorstore()  # loaded once, before the first yield

    for _, row in df.iterrows():
        results = store.similarity_search_with_score(row["question"], k=k)
        yield row, results


if __name__ == "__main__":
    for row, results in run_retrieval():
        print(f"\nQ{row['id']} [{row['type']}]: {row['question']}")
        for doc,score in results:
            m = doc.metadata
            print(f" {score:.4f} {m['document_id']} | page {m['page'] + 1} | {m['chunk_id']}")
            print(f"    {doc.page_content[:200]!r}")