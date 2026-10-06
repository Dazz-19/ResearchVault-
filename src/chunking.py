from src.ingestion import load_all_papers
from langchain.text_splitter import RecursiveCharacterTextSplitter




def chunk_papers(docs):
    """Split a list of page Documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400)
    chunks = splitter.split_documents(docs)
    #creating unique chunk ids for each chunk
    notebook={}
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        page_num = chunk.metadata["page"]
        count=notebook.get((doc_id, page_num), 0)
        count+=1
        notebook[(doc_id, page_num)] = count
        chunk.metadata["chunk_id"] = f"{doc_id}_p{page_num+1}_c{count}"


    return chunks



if __name__ == "__main__":
    docs = load_all_papers()
    chunks = chunk_papers(docs)
    if len(chunks)==len(set(chunk.metadata["chunk_id"] for chunk in chunks)):
        print("All chunk IDs are unique.")
    for chunk in chunks:
        print(chunk.metadata["chunk_id"])
    # print(f"\nTotal chunks: {len(chunks)}")
   