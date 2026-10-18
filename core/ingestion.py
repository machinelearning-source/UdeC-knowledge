import os
import sys
import tempfile

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import FastEmbedEmbeddings

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def run_ingestion():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    documents_path = os.path.join(base_path, "data", "documents")
    vector_store_path = os.path.join(base_path, "vector_store")
    model_cache_path = os.path.join(tempfile.gettempdir(), "fastembed_cache")

    print(f"[PROCESS] Starting to read PDFs in: {documents_path}")

    if not os.path.exists(documents_path):
        print("[WARNING] The data/documents folder does not exist. Create the document structure before running ingestion.")
        return False

    pdf_files = [
        name
        for name in sorted(os.listdir(documents_path))
        if name.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("[WARNING] The data/documents folder is empty. Please add at least one institutional PDF file.")
        return False

    loader = PyPDFDirectoryLoader(documents_path, glob="**/*.pdf")
    loaded_documents = loader.load()
    print(f"[OK] Successfully read {len(loaded_documents)} pages from documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(loaded_documents)
    print(f"[PROCESS] The text has been split into {len(text_chunks)} indexable chunks.")

    os.makedirs(vector_store_path, exist_ok=True)

    print("[PROCESS] Building the vector database with FAISS...")
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, cache_dir=model_cache_path)
    vector_db = FAISS.from_documents(text_chunks, embeddings)

    vector_db.save_local(vector_store_path)
    print(f"[SUCCESS] Vector database persisted correctly in: {vector_store_path}")
    return True


if __name__ == "__main__":
    run_ingestion()
