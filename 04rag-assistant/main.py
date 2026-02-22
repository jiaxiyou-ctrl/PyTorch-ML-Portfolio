"""RAG Assistant — Full pipeline entry point: load, split, embed, retrieve, generate."""

import warnings
import os
import time

warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from step1_load import load_document
from step2_split import split_documents
from step3_embed import create_vector_store
from step4_retrieve import get_retriever
from step5_generate import create_llm, create_qa_chain, clean_answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

DEMO_QUESTIONS = [
    "What types of zombies are there?",
    "What is the best weapon for close combat?",
    "How do I purify water?",
    "What should I do if someone gets bitten?",
]


def main():
    start_time = time.time()

    print("=" * 60)
    print("  🧟 RAG Assistant: Zombie Survival Guide")
    print("=" * 60)

    # --- Load ---
    print("\n📥 Loading documents...")
    file_path = os.path.join(DATA_DIR, "sample.txt")
    docs = load_document(file_path)

    # --- Split ---
    print("\n✂️  Splitting into chunks...")
    chunks = split_documents(docs)
    print("\n   Preview (first 3):")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:80].replace("\n", "\\n")
        print(f'   📦 Chunk {i+1} ({len(chunk.page_content)} chars): "{preview}..."')
    remaining = len(chunks) - 3
    if remaining > 0:
        print(f"   ... and {remaining} more chunks")

    # --- Embed ---
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print(f"\n📦 Vector store already exists at {DB_DIR}, skipping embedding.")
    else:
        print("\n🧠 Creating vector store...")
        create_vector_store(chunks, DB_DIR)

    # --- Retriever ---
    print("\n🔗 Loading retriever...")
    retriever = get_retriever(DB_DIR)

    # --- LLM ---
    print("\n🤖 Loading LLM...")
    llm = create_llm()

    # --- QA Chain ---
    print("\n🔗 Creating QA chain...")
    qa_chain = create_qa_chain(llm, retriever)

    # --- Demo queries ---
    total = len(DEMO_QUESTIONS)
    for idx, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'─'*60}")
        print(f"--- Query {idx}/{total} ---")
        print(f"❓ Q: {question}")
        result = qa_chain.invoke({"query": question})
        answer = clean_answer(result["result"])
        sources = len(result.get("source_documents", []))
        print(f"🤖 A: {answer}")
        print(f"📚 Sources: {sources} chunks")

    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Done! Total execution time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
