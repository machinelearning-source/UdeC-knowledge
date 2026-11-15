"""KnowBot retrieval pieces needed by the re-ranker.

Copied from the KnowBot app (core/pipeline.py) so this project runs on its own:
the FAISS index, the BM25 keyword index, and the exact calendar and article
lookups. Nothing here calls an external API.
"""

import math
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np

from .calendar_names import calendar_name_in
from .embeddings import DEFAULT_MODEL, FastEmbedEmbeddings, default_cache_dir

CALENDAR_PREFIX = "ACADEMIC CALENDAR"
CALENDAR_HINTS = r"\b(calendar|calendario|when|cu[aá]ndo|date|dates|fecha|fechas|deadline|plazo|schedule)\b"
POSTGRADUATE_SUFFIX = "(POSGRADOS)"
ARTICLE_PATTERN = re.compile(r"\b(?:article|art[ií]culo|art\.)s?\s+(\d{1,3})\b", re.IGNORECASE)
ARTICLE_CHUNKS_PER_NUMBER = 2
KEYWORD_STOPWORDS = set(
    "the of and to in a an for is are what which how who when does do on at by with from as be or that this "
    "its their el la los las de del y en un una por para con que se es su al lo".split()
)
VECTOR_STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store")


def file_name(source):
    return os.path.basename(source or "")


def normalize(text):
    """Uppercase and strip accents so 'Línea' and 'LINEA' compare equal."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip().upper()


def keyword_tokens(text):
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return [token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in KEYWORD_STOPWORDS]


def is_calendar_chunk(doc):
    return file_name(doc.metadata.get("source", "")).upper().startswith(CALENDAR_PREFIX)


def looks_like_calendar_question(question):
    return re.search(CALENDAR_HINTS, question, re.IGNORECASE) is not None


class KeywordIndex:
    """BM25 keyword search over the same fragments as the FAISS index."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1, self.b = k1, b
        self.term_counts = [Counter(keyword_tokens(doc.page_content)) for doc in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency = Counter(term for counts in self.term_counts for term in counts)

    def scores(self, query):
        """BM25 score of every fragment for the query (0 when no word matches)."""
        terms = set(keyword_tokens(query))
        total = len(self.documents)
        result = np.zeros(total)
        for index, counts in enumerate(self.term_counts):
            for term in terms & counts.keys():
                frequency = self.document_frequency[term]
                idf = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
                tf = counts[term]
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                result[index] += idf * tf * (self.k1 + 1) / (tf + norm)
        return result


def calendar_activity_matches(chunks, question):
    """Indexes of the calendar lines whose activity the question names (exact official name)."""
    official_name = calendar_name_in(question)
    if not official_name:
        return set()
    name = normalize(official_name)
    section = "POSGRADOS" if name.endswith(POSTGRADUATE_SUFFIX) else "PREGRADO"
    name = name.replace(POSTGRADUATE_SUFFIX, "").strip()
    activities = []
    for index, doc in enumerate(chunks):
        if is_calendar_chunk(doc):
            parts = normalize(doc.page_content).split(" — ")
            if len(parts) >= 2 and f"({section})" in parts[0]:
                activities.append((parts[1], index))
    starts_with = {index for activity, index in activities if activity.startswith(name)}
    return starts_with or {index for activity, index in activities if name in activity}


def article_matches(chunks, question):
    """Indexes of the chunks with the articles a question mentions ('Article 35' -> 'ARTÍCULO 35')."""
    found = set()
    for number in dict.fromkeys(ARTICLE_PATTERN.findall(question)):
        heading = re.compile(rf"\bARTICULO\s+{number}\b(?!\s*\d)")
        matches = [index for index, doc in enumerate(chunks) if heading.search(normalize(doc.page_content))]
        found.update(matches[:ARTICLE_CHUNKS_PER_NUMBER])
    return found


class Corpus:
    """The indexed fragments, their embedding vectors and the BM25 index, loaded once."""

    def __init__(self, vector_store_path=VECTOR_STORE):
        from langchain_community.vectorstores import FAISS

        self.embeddings = FastEmbedEmbeddings(model_name=DEFAULT_MODEL, cache_dir=default_cache_dir())
        store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        ids = [store.index_to_docstore_id[i] for i in range(store.index.ntotal)]
        self.chunks = [store.docstore.search(doc_id) for doc_id in ids]
        vectors = np.vstack([store.index.reconstruct(i) for i in range(store.index.ntotal)])
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.keyword_index = KeywordIndex(self.chunks)
        self.max_page = Counter()
        for doc in self.chunks:
            name = file_name(doc.metadata.get("source", ""))
            self.max_page[name] = max(self.max_page[name], doc.metadata.get("page", 0) + 1)

    def query_vector(self, question):
        vector = np.asarray(self.embeddings.embed_query(question))
        return vector / np.linalg.norm(vector)


@lru_cache(maxsize=1)
def load_corpus():
    import faiss

    faiss.omp_set_num_threads(1)  # one thread: the search covers only 290 vectors
    return Corpus()
