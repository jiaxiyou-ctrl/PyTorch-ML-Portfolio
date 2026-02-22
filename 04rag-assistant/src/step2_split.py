"""Step 2: Text Splitting — Break documents into smaller, overlapping chunks."""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")


def split_documents(documents):
    """Split documents into chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)

    print(f"✅ Documents split successfully!")
    print(f"📦 {len(documents)} document(s) split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        print(f"\n📦 Chunk {i+1} ({len(chunk.page_content)} chars):")
        print(f"   {chunk.page_content}")
        print(f"   {'─'*40}")

    return chunks


if __name__ == "__main__":
    from step1_load import load_document
    file_path = os.path.join(DATA_DIR, "sample.txt")
    docs = load_document(file_path)
    chunks = split_documents(docs)
