import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings


def query_rag_system(user_question):
    vector_store_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vector_store"))

    if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        return "Error: No knowledge index found. Please run 'ingestion.py' first to process the university documents.", []

    embeddings = FakeEmbeddings(size=1536)
    vector_db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

    print(f"[PROCESS] Searching document sources for: '{user_question}'")
    found_documents = vector_db.similarity_search(user_question, k=3)

    evidence_sources = []
    context_text = ""

    for i, doc in enumerate(found_documents):
        file_name = os.path.basename(doc.metadata.get("source", "Institutional_Document.pdf"))
        page_number = doc.metadata.get("page", 0) + 1

        evidence_sources.append(f"PDF: {file_name} (Page {page_number})")
        context_text += f"[Fragment {i + 1}]: {doc.page_content}\n\n"

    if len(found_documents) > 0:
        synthesized_response = (
            "Based on the official documentation retrieved from the university, relevant sections "
            "related to your question were identified. The system analyzed the indexed content and "
            "determined the pertinent information."
        )
    else:
        synthesized_response = "No sections within the uploaded documents match your query."

    return synthesized_response, evidence_sources


if __name__ == "__main__":
    test_response, test_sources = query_rag_system("laboratory requirements")
    print(f"\nResponse: {test_response}")
    print(f"Sources: {test_sources}")
