"""Step 3: Embedding & Storage — Vectorize chunks and persist to ChromaDB."""

import os
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.CRITICAL)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")


def create_vector_store(chunks, presist_dir=DB_DIR):
    """Embed document chunks and store them in a Chroma vector database."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=presist_dir,
    )

    print(f"✅ Vector database created successfully!")
    print(f"📊 Stored {vector_store._collection.count()} vectors")
    return vector_store


if __name__ == "__main__":
    from step1_load import load_document
    from step2_split import split_documents

    file_path = os.path.join(DATA_DIR, "sample.txt")
    docs = load_document(file_path)
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks, DB_DIR)

    query = "What types of zombies are there?"
    results = vector_store.similarity_search(query, k=2)
    print(f"\n🔍 Search results for '{query}':")
    for i, doc in enumerate(results):
        print(f"   Result {i+1}: {doc.page_content}")
