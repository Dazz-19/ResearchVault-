from src.embeddings import get_embeddings
from langchain_chroma import Chroma 
import pandas as pd

DB="chroma_db"

def create_vectorstore(chunksp):
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
    from src.chunking import chunk_papers
    from src.ingestion import load_all_papers
    docs = load_all_papers()
    chunks = chunk_papers(docs,500,200)
    vector_store1=load_vectorstore(1000,400)
    print(vector_store1._collection.count())
    vs=create_vectorstore(chunks,500,200)
    vs2=load_vectorstore(500,200)
    print(vs2._collection.count())
    