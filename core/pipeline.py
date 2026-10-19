import os
import re
import sys
import tempfile
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import FastEmbedEmbeddings

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are the institutional knowledge assistant of this university.

Rules:
- Answer ONLY using the information contained in the official document fragments provided below.
- Always write the whole answer in the same language as the user's question, even when the document fragments are written in another language.
- The fragments are usually written in Spanish. That must NEVER change your language: if the question is in English, every word of your answer must be in English; if the question is in Spanish, answer fully in Spanish.
- Be clear, brief and use plain language.
- If the fragments do NOT contain enough information to answer, say so explicitly: tell the user the information was not found in the consulted official documents. Never invent information or use external knowledge.
- Always end with a short list of the sources used (document name and page)."""

SPANISH_MARKERS = r"[áéíóúñ¿¡]|\b(qué|cuál|cuáles|cómo|dónde|cuándo|cuánto|cuántos|es|son|debo|puedo|universidad|estudiante|estudiantes)\b"
ENGLISH_MARKERS = r"\b(what|which|how|why|when|where|who|is|are|does|do|can|the|university|student|students)\b"


def detect_question_language(question):
    spanish_hits = len(re.findall(SPANISH_MARKERS, question, re.IGNORECASE))
    english_hits = len(re.findall(ENGLISH_MARKERS, question, re.IGNORECASE))
    return "Spanish" if spanish_hits > english_hits else "English"


@lru_cache(maxsize=1)
def load_vector_store(vector_store_path, model_cache_path):
    """Load the embedding model and FAISS index only once per process.

    Loading the ONNX embedding model is expensive (tens of seconds on a
    low-CPU host), so it must never happen on every question.
    """
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, cache_dir=model_cache_path)
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)


def query_rag_system(user_question):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vector_store_path = os.path.join(base_path, "vector_store")
    model_cache_path = os.path.join(tempfile.gettempdir(), "fastembed_cache")

    load_dotenv(os.path.join(base_path, ".env"))

    if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        return "Error: No knowledge index found. Please run 'python core/ingestion.py' first to process the university documents.", []

    print(f"[PROCESS] Searching document sources for: '{user_question}'")
    vector_db = load_vector_store(vector_store_path, model_cache_path)
    found_documents = vector_db.similarity_search(user_question, k=4)

    if len(found_documents) == 0:
        return "No sections within the uploaded documents match your query.", []

    evidence_sources = []
    context_text = ""

    for i, doc in enumerate(found_documents):
        file_name = os.path.basename(doc.metadata.get("source", "Institutional_Document.pdf"))
        page_number = doc.metadata.get("page", 0) + 1

        evidence_sources.append(f"PDF: {file_name} (Page {page_number})")
        context_text += f"[Fragment {i + 1}] (Source: {file_name}, page {page_number}):\n{doc.page_content}\n\n"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not configured. Set it in the Render environment variables (or in a local .env file) and restart the app.", []

    try:
        llm = ChatGroq(temperature=0, model_name=LLM_MODEL, api_key=api_key)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Write your entire answer in {language}.\n\nOfficial document fragments:\n\n{context}\n\nUser question: {question}",
                ),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        synthesized_response = chain.invoke(
            {
                "context": context_text,
                "question": user_question,
                "language": detect_question_language(user_question),
            }
        )
    except Exception as exc:
        return f"Error: the AI service could not answer ({exc})", []

    if not synthesized_response.strip():
        return "Error: the AI service returned an empty answer.", []

    return synthesized_response, evidence_sources


if __name__ == "__main__":
    test_response, test_sources = query_rag_system("What are the requirements for an academic procedure?")
    print(f"\nResponse: {test_response}")
    print(f"Sources: {test_sources}")
