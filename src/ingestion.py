from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path):
    """Load one PDF and attach metadata to every page."""
    path = Path(path)
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    for page in pages:
        page.metadata["document_id"] = path.stem
        page.metadata["title"] = path.stem      # filename for now, improve later
        page.metadata["filename"] = path.name

    return pages


def load_all_papers(folder="data"):
    """Load every PDF in a folder into one list of page Documents."""
    all_docs = []

    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        try:
            pages = load_pdf(pdf_path)
            all_docs.extend(pages)
            print(f"Loaded {pdf_path.name}: {len(pages)} pages")
        except Exception as e:
            print(f"Skipped {pdf_path.name}: {e}")

    return all_docs


if __name__ == "__main__":
    docs = load_all_papers()
    print(f"\nTotal pages: {len(docs)}")
    if docs:
        print("\n" + str(docs[0].metadata))