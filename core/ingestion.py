import os
import re
import sys

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import FastEmbedEmbeddings, default_cache_dir

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CALENDAR_PREFIX = "ACADEMIC CALENDAR"

CALENDAR_MONTHS = [
    ("December 2025", 3),
    ("January 2026", 4),
    ("February 2026", 5),
    ("March 2026", 6),
    ("April 2026", 7),
    ("May 2026", 8),
    ("June 2026", 9),
    ("July 2026", 10),
    ("August 2026", 11),
    ("September 2026", 12),
    ("October 2026", 13),
    ("November 2026", 14),
    ("December 2026", 15),
    ("January 2027", 16),
]

DATE_PATTERN = re.compile(r"^\d{1,2}(-\d{1,2})?$")


def _clean(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def calendar_page_to_lines(page):
    """Read one academic-calendar page as aligned activity lines.

    The calendar is a wide table (one column per month). Plain PDF text
    extraction loses the column that every date belongs to, which is why the
    dates are read from the table cells instead. Every activity becomes its
    own line, so retrieval returns the exact dates for one activity.
    """
    tables = page.extract_tables()
    lines = []
    section = ""

    for table in tables:
        for row in table:
            cells = [_clean(cell) for cell in row]
            if len(cells) < 17:
                continue

            header = cells[1].upper()
            if header.startswith("ACTIVIDADES"):
                section = header.replace("ACTIVIDADES", "").strip() or "PREGRADO"
                continue

            dates = [
                f"{name}: {cells[index]}"
                for name, index in CALENDAR_MONTHS
                if DATE_PATTERN.match(cells[index])
            ]
            if not dates:
                continue

            activity = cells[1] or (lines[-1]["activity"] if lines else "")
            if not activity:
                continue

            label = f"{activity} {cells[2]}".strip()
            lines.append(
                {
                    "section": section,
                    "activity": activity,
                    "text": f"2026 academic calendar ({section}) — {label} — {'; '.join(dates)}",
                }
            )

    return lines


def load_calendar_documents(documents_path):
    """Load calendar PDFs with table-aware extraction instead of plain text."""
    import pdfplumber

    documents = []
    for name in sorted(os.listdir(documents_path)):
        if not name.lower().endswith(".pdf") or not name.upper().startswith(CALENDAR_PREFIX):
            continue

        with pdfplumber.open(os.path.join(documents_path, name)) as pdf:
            for page_number, page in enumerate(pdf.pages):
                for line in calendar_page_to_lines(page):
                    documents.append(
                        Document(
                            page_content=line["text"],
                            metadata={"source": name, "page": page_number},
                        )
                    )

    if documents:
        print(f"[OK] Calendar rebuilt into {len(documents)} aligned activities.")
    return documents


def is_calendar(document):
    source = os.path.basename(document.metadata.get("source", ""))
    return source.upper().startswith(CALENDAR_PREFIX)


def run_ingestion():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    documents_path = os.path.join(base_path, "data", "documents")
    vector_store_path = os.path.join(base_path, "vector_store")
    model_cache_path = default_cache_dir()

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
    loaded_documents = [document for document in loader.load() if not is_calendar(document)]
    loaded_documents += load_calendar_documents(documents_path)
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
