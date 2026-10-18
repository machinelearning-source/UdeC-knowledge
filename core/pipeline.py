import os
import sys
import tempfile

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import FastEmbedEmbeddings

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are the institutional knowledge assistant of Universidad de Cundinamarca (UdeC).

Rules:
- Answer ONLY using the information contained in the official document fragments provided below.
- Answer in the same language the user used in their question.
- Be clear, brief and use plain language.
- If the fragments do NOT contain enough information to answer, say so explicitly: tell the user the information was not found in the consulted official documents. Never invent information or use external knowledge.
- Always end with a short list of the sources used (document name and page)."""


def query_rag_system(user_question):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vector_store_path = os.path.join(base_path, "vector_store")
    model_cache_path = os.path.join(tempfile.gettempdir(), "fastembed_cache")

    load_dotenv(os.path.join(base_path, ".env"))

    if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        return "Error: No knowledge index found. Please run 'python core/ingestion.py' first to process the university documents.", []

    print(f"[PROCESS] Searching document sources for: '{user_question}'")
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, cache_dir=model_cache_path)
    vector_db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
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
                    "Official document fragments:\n\n{context}\n\nUser question: {question}",
                ),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        synthesized_response = chain.invoke({"context": context_text, "question": user_question})
    except Exception as exc:
        return f"Error: the AI service could not answer ({exc})", []

    if not synthesized_response.strip():
        return "Error: the AI service returned an empty answer.", []

    return synthesized_response, evidence_sources


if __name__ == "__main__":
    test_response, test_sources = query_rag_system("What are the requirements for an academic procedure?")
    print(f"\nResponse: {test_response}")
    print(f"Sources: {test_sources}")
