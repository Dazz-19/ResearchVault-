from src.embeddings import get_embeddings
from langchain_chroma import Chroma 

DB="chroma_db"

def create_vectorstore(chunks):
    ids=[chunk.metadata["chunk_id"] for chunk in chunks]
    model=get_embeddings()
    """Create a vectorstore from the chunks."""
    vectorstore = Chroma.from_documents(documents=chunks, embedding=model,persist_directory=DB,ids=ids)
    return vectorstore

def load_vectorstore():
    """Load the vectorstore from disk."""
    model=get_embeddings()
    vectorstore = Chroma(persist_directory=DB, embedding_function=model)
    return vectorstore



if __name__ == "__main__":
    # from src.chunking import chunk_papers
    # from src.ingestion import load_all_papers
    # docs = load_all_papers()
    # chunks = chunk_papers(docs)
    # vectors = create_vectorstore(chunks)

    vectorstore = load_vectorstore()
    vectors = vectorstore.similarity_search("What is a transformer model?", k=1)
    print(vectors[0].page_content)
    print(vectors[0].metadata["document_id"])
    print(vectors[0].metadata["chunk_id"])
    print(vectors[0].metadata["page"])