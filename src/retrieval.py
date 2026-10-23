from src.vectorstore import load_vectorstore
from sentence_transformers import CrossEncoder

# Both are loaded on first use, not when this module is imported.
Store = None
reranker = None

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_store():
    """Load the vector store once and reuse it."""
    global Store
    if Store is None:
        Store = load_vectorstore()
    return Store


def get_retrieval_results(question, k=10, filter_paper=None):
    """Plain embedding search. Returns a list of (Document, score).

    The score is a DISTANCE: lower is better.
    """
    store = _get_store()
    if filter_paper:
        return store.similarity_search_with_score(
            question, k=k, filter={"document_id": filter_paper}
        )
    return store.similarity_search_with_score(question, k=k)


def retrieve_with_rerank(question, k=5, fetch_k=50, filter_paper=None):
    """Embedding search for fetch_k candidates, then rerank and keep the best k.

    Returns a list of (Document, score) in ranked order.
    The score is a reranker score: HIGHER is better (the opposite of
    get_retrieval_results, which returns a distance).
    """
    global reranker
    if reranker is None:
        reranker = CrossEncoder(RERANKER_MODEL)

    store = _get_store()
    if filter_paper:
        candidates = store.similarity_search(
            question, k=fetch_k, filter={"document_id": filter_paper}
        )
    else:
        candidates = store.similarity_search(question, k=fetch_k)

    if not candidates:
        return []

    pairs = [(question, d.page_content) for d in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:k]]