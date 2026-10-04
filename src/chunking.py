from src.ingestion import load_all_papers
from langchain.text_splitter import RecursiveCharacterTextSplitter

docs = load_all_papers()


def chunk_papers(docs):
    """Split a list of page Documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks



if __name__ == "__main__":
    chunks = chunk_papers(docs)
    print(f"\nTotal chunks: {len(chunks)}")
   