"""Step 1: Document Loading — Read raw text files into LangChain Documents."""

import os
from langchain_community.document_loaders import TextLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")


def load_document(file_path):
    """Load a text file and return a list of Document objects."""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print(f"✅ Document loaded successfully!")
    print(f"📄 Number of documents: {len(documents)}")
    print(f"📝 Content preview (first 200 chars):\n{documents[0].page_content[:200]}")

    return documents


if __name__ == "__main__":
    file_path = os.path.join(DATA_DIR, "sample.txt")
    docs = load_document(file_path)
