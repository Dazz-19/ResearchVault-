
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """Get model for embeddings for the documents."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if __name__ == "__main__":
    model = get_embeddings()
    vectors=model.embed_query("This is a test query.")
    print(len(vectors))