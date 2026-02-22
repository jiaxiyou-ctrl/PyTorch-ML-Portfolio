"""Step 5: Generation — Feed retrieved context to a local LLM for answers."""

import warnings
import os
warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")


def create_llm():
    """Load TinyLlama as a local text-generation pipeline."""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"⏳ Loading model: {model_name}...")

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        device=-1,
        pipeline_kwargs={
            "max_new_tokens": 128,
            "temperature": 0.1,
            "do_sample": True,
            "return_full_text": False,
        },
    )

    print("✅ Model loaded successfully!")
    return llm


def create_qa_chain(llm, retriever):
    """Build a RetrievalQA chain connecting the retriever to the LLM."""
    prompt_template = """<|system|>
You are a zombie apocalypse survival expert. Answer based ONLY on the reference materials. Be concise. If unsure, say "Not in my survival guide."</s>
<|user|>
Reference materials:
{context}

Question: {question}</s>
<|assistant|>
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


def clean_answer(raw_answer):
    """Strip template artifacts and deduplicate lines from model output."""
    answer = raw_answer.strip()

    junk_phrases = [
        "<|user|>", "<|system|>",
        "Reference materials:", "Question:", "Answer:",
    ]
    for phrase in junk_phrases:
        if phrase in answer:
            answer = answer.split(phrase)[-1].strip()

    lines = answer.split("\n")
    seen_lines = set()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped not in seen_lines:
            seen_lines.add(stripped)
            clean_lines.append(line)
    answer = "\n".join(clean_lines)

    return answer


if __name__ == "__main__":
    from step4_retrieve import get_retriever

    print("📚 Loading retriever...")
    retriever = get_retriever()

    llm = create_llm()

    print("🔗 Creating QA chain...")
    qa_chain = create_qa_chain(llm, retriever)

    questions = [
        "What types of zombies are there?",
        "What is the best weapon for close combat?",
        "How do I purify water?",
        "What should I do if someone gets bitten?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ {q}")
        result = qa_chain.invoke({"query": q})
        answer = clean_answer(result["result"])
        print(f"🤖 {answer}")
