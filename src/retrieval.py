from src.vectorstore import load_vectorstore
global Store
Store=None


def get_retrieval_results(question,k=5,filter_paper=None):
    """Return the top-k retrieval results for a given question."""
    global Store
    if Store is None:
        Store=load_vectorstore()
    if filter_paper:
        results=Store.similarity_search_with_score(question,k=k,filter={"document_id":filter_paper})
    else:               
        results=Store.similarity_search_with_score(question,k=k)
    return results 





if __name__ == "__main__":
    question = "What is a transformer model?"
    results = get_retrieval_results(question, k=5)
    for doc, score in results:
        m = doc.metadata
        print(f"Score: {score:.4f} | Document: {m['document_id']} | Page: {m['page'] + 1} | Chunk: {m['chunk_id']}")
        print(f"Content: {doc.page_content[:200]!r}\n")