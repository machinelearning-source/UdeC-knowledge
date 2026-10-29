import math
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import FastEmbedEmbeddings, default_cache_dir
from core.calendar_names import calendar_name_in

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Groq limits are per model: when the main model runs out of its daily tokens the
# answer is written by this smaller model, and follow-ups are always rewritten by it.
FALLBACK_LLM_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
# Short reasoning roughly halves the tokens of a gpt-oss answer.
REASONING_EFFORT = "low"
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 1

SYSTEM_PROMPT = """You are the institutional knowledge assistant of this university.

Rules:
- Answer ONLY using the information contained in the official document fragments provided below.
- Always write the whole answer in English, even when the user's question or the document fragments are written in Spanish.
- The fragments are usually written in Spanish. That must NEVER change your language: every word of your answer must be in English.
- Translate Spanish institutional terms, course names, calendar activity names and document names into English. Never include Spanish words or Spanish glosses in parentheses: for every Spanish term in the question or in the fragments, write only its English equivalent.
- Academic calendar fragments look like '2026 academic calendar (PREGRADO) — ACTIVITY NAME INICIA — March 2026: 30; September 2026: 1'. PREGRADO means undergraduate and POSGRADOS postgraduate (in an English answer write only 'undergraduate' or 'postgraduate', never PREGRADO or POSGRADOS); INICIA marks the start and TERMINA the end of the activity; each 'Month Year: day' (or 'day-day') is one date or date range, and several dates mean the activity happens in each academic period. Combine the INICIA and TERMINA lines of the same activity into one period per semester.
- Be clear, brief and use plain language.
- If the fragments do NOT contain enough information to answer, say so explicitly: tell the user the information was not found in the consulted official documents. Never invent information or use external knowledge.
- Never repeat the same term in parentheses (write 'admission test', never 'admission test (Admission Test)').
- Cite the fragments you use with their number in square brackets right after the sentence they support, for example [2] or [1][4]. Use only fragment numbers that appear below.
- Do NOT add a 'Sources' section or a list of documents and pages at the end: the app already shows the cited sources below every answer."""

FOLLOW_UP_PROMPT = """Rewrite the user's last question as one standalone question in English that can be understood without the conversation.
Keep every detail of the last question and take the missing subject (activity, course, student situation, document) from the conversation.
If the last question is already standalone, return it translated to English and otherwise unchanged.
Return only the rewritten question, with no explanation."""

CALENDAR_PREFIX = "ACADEMIC CALENDAR"
CALENDAR_HINTS = r"\b(calendar|calendario|when|cu[aá]ndo|date|dates|fecha|fechas|deadline|plazo|schedule)\b"
FOLLOW_UP_HINTS = (
    r"\b(it|its|that|this|those|these|they|them|there|also|what about|how about|and for|"
    r"same|instead|eso|esa|ese|tambi[eé]n|y para|y si)\b"
)
FOLLOW_UP_MAX_WORDS = 7
POSTGRADUATE_SUFFIX = "(POSGRADOS)"
CITATION_PATTERN = re.compile(r"[\[【](\d{1,2})(?:†[^\]】]*)?[\]】]")
ARTICLE_PATTERN = re.compile(r"\b(?:article|art[ií]culo|art\.)s?\s+(\d{1,3})\b", re.IGNORECASE)
ARTICLE_CHUNKS_PER_NUMBER = 2
KEYWORD_HITS = 2
MAX_FRAGMENTS = 10
KEYWORD_STOPWORDS = set(
    "the of and to in a an for is are what which how who when does do on at by with from as be or that this "
    "its their el la los las de del y en un una por para con que se es su al lo".split()
)
MAX_HISTORY_MESSAGES = 4
MAX_HISTORY_CHARS = 1200
FALLBACK_SOURCES = 4

ANSWER_LANGUAGE = "English"
NOT_FOUND_MARKERS = ("not found", "not contain", "do not provide", "does not provide", "no information")


@dataclass
class RagRequest:
    """Everything the answer step needs: the standalone question and its fragments."""

    question: str
    context_text: str = ""
    fragment_sources: list = field(default_factory=list)
    error: str = ""


def _base_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _vector_store_path():
    return os.path.join(_base_path(), "vector_store")


def _file_name(source):
    """Return the file name of a source path written with / or \\ separators."""
    return re.split(r"[\\/]", source or "")[-1] or "Institutional_Document.pdf"


def _llm(model=LLM_MODEL):
    load_dotenv(os.path.join(_base_path(), ".env"))
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    extra = {"reasoning_effort": REASONING_EFFORT} if model.startswith("openai/gpt-oss") else {}
    # A short timeout and one retry: when Groq is rate limited it can ask to wait
    # minutes, and the app must show the error instead of freezing.
    return ChatGroq(
        temperature=0, model_name=model, api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES, model_kwargs=extra,
    )


def _is_daily_limit(exc):
    return "tokens per day" in str(exc) or "requests per day" in str(exc)


@lru_cache(maxsize=1)
def load_vector_store(vector_store_path, model_cache_path):
    """Load the embedding model and FAISS index only once per process.

    Loading the ONNX embedding model is expensive (tens of seconds on a
    low-CPU host), so it must never happen on every question.
    """
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, cache_dir=model_cache_path)
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)


def _normalize(text):
    """Uppercase and strip accents so 'Línea' and 'LINEA' compare equal."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip().upper()


def _is_calendar_document(doc):
    return _file_name(doc.metadata.get("source", "")).upper().startswith(CALENDAR_PREFIX)


def _calendar_documents(vector_db):
    return [doc for doc in vector_db.docstore._dict.values() if _is_calendar_document(doc)]


def find_calendar_activity(vector_db, official_name):
    """Return the calendar lines of one activity, matched by its official name.

    Every calendar line looks like
    '2026 academic calendar (PREGRADO) — INSCRIPCIONES EN LÍNEA INICIA — March 2026: 30'.
    The 71 lines are very similar to each other, so semantic search alone often
    returns the wrong activity; an exact name match is reliable.
    """
    name = _normalize(official_name)
    section = "POSGRADOS" if name.endswith(POSTGRADUATE_SUFFIX) else "PREGRADO"
    name = name.replace(POSTGRADUATE_SUFFIX, "").strip()

    activities = []
    for doc in _calendar_documents(vector_db):
        parts = _normalize(doc.page_content).split(" — ")
        if len(parts) >= 2 and f"({section})" in parts[0]:
            activities.append((parts[1], doc))

    starts_with = [doc for activity, doc in activities if activity.startswith(name)]
    return starts_with or [doc for activity, doc in activities if name in activity]


def find_articles(vector_db, question):
    """Return the chunks that contain the articles a question mentions ('Article 35' -> 'ARTÍCULO 35')."""
    found = []
    for number in dict.fromkeys(ARTICLE_PATTERN.findall(question)):
        heading = re.compile(rf"\bARTICULO\s+{number}\b(?!\s*\d)")
        matches = [doc for doc in vector_db.docstore._dict.values() if heading.search(_normalize(doc.page_content))]
        found += matches[:ARTICLE_CHUNKS_PER_NUMBER]
    return found


def _looks_like_calendar_question(question):
    return re.search(CALENDAR_HINTS, question, re.IGNORECASE) is not None


def _keyword_tokens(text):
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return [token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in KEYWORD_STOPWORDS]


class KeywordIndex:
    """BM25 keyword search over the same fragments as the FAISS index.

    Vectors capture meaning but miss exact words; the validation questions often
    quote Spanish names ("Comisión Reglamento Estudiantil") that a keyword search
    finds directly. It lives in memory and is built from the docstore, so the
    FAISS index does not need to be rebuilt.
    """

    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1, self.b = k1, b
        self.term_counts = [Counter(_keyword_tokens(doc.page_content)) for doc in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency = Counter(term for counts in self.term_counts for term in counts)

    def search(self, query, k):
        terms = set(_keyword_tokens(query))
        total = len(self.documents)
        scored = []
        for index, counts in enumerate(self.term_counts):
            score = 0.0
            for term in terms & counts.keys():
                frequency = self.document_frequency[term]
                idf = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
                tf = counts[term]
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                score += idf * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scored.append((score, index))
        return [self.documents[index] for _, index in sorted(scored, reverse=True)[:k]]


_keyword_indexes = {}


def _keyword_index(vector_db):
    if id(vector_db) not in _keyword_indexes:
        _keyword_indexes[id(vector_db)] = KeywordIndex(list(vector_db.docstore._dict.values()))
    return _keyword_indexes[id(vector_db)]


def _unique(documents):
    unique_documents, seen_contents = [], set()
    for doc in documents:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_documents.append(doc)
    return unique_documents[:MAX_FRAGMENTS]


def retrieve_documents(vector_db, search_query, calendar_name=None):
    """Combine exact matches, the MMR search, a keyword (BM25) search and a calendar-only search.

    The calendar-only results go last: questions with 'when' or 'date' are not
    always about the calendar, and they must not push the right regulation out.
    The keyword hits go right after the first result (hybrid retrieval): on the
    validation set this raised the right page (±1) in the top 6 from 88% to 93%.
    """
    found_documents = []
    if calendar_name:
        found_documents += find_calendar_activity(vector_db, calendar_name)
    found_documents += find_articles(vector_db, search_query)

    found_documents += vector_db.max_marginal_relevance_search(
        search_query, k=6, fetch_k=40, lambda_mult=0.6
    )

    if calendar_name or _looks_like_calendar_question(search_query):
        found_documents += vector_db.similarity_search(
            search_query, k=4, fetch_k=80, filter=lambda metadata: _file_name(
                metadata.get("source", "")
            ).upper().startswith(CALENDAR_PREFIX)
        )

    found_documents = _unique(found_documents)
    keyword_hits = _keyword_index(vector_db).search(search_query, KEYWORD_HITS)
    return _unique(found_documents[:1] + keyword_hits + found_documents[1:])


def warm_up():
    """Preload the embedding model and the FAISS index into memory."""
    if os.path.exists(os.path.join(_vector_store_path(), "index.faiss")):
        load_vector_store(_vector_store_path(), default_cache_dir())


def knowledge_stats():
    """Count the indexed documents and sections without loading the embedding model."""
    index_path = os.path.join(_vector_store_path(), "index.pkl")
    if not os.path.exists(index_path):
        return {"documents": 0, "sections": 0}
    with open(index_path, "rb") as index_file:
        docstore, _ = pickle.load(index_file)
    documents = list(docstore._dict.values())
    return {
        "documents": len({_file_name(doc.metadata.get("source", "")) for doc in documents}),
        "sections": len(documents),
    }


def _history_text(history):
    lines = []
    for message in (history or [])[-MAX_HISTORY_MESSAGES:]:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {CITATION_PATTERN.sub('', message['content'])[:MAX_HISTORY_CHARS]}")
    return "\n".join(lines)


def _looks_like_follow_up(question):
    words = question.split()
    return len(words) <= FOLLOW_UP_MAX_WORDS or re.search(FOLLOW_UP_HINTS, question, re.IGNORECASE) is not None


def standalone_question(user_question, history, llm):
    """Turn a follow-up like 'and for postgraduate?' into a full question using the chat history."""
    if not history or not _looks_like_follow_up(user_question):
        return user_question
    prompt = ChatPromptTemplate.from_messages(
        [("system", FOLLOW_UP_PROMPT), ("human", "Conversation:\n{history}\n\nLast question: {question}")]
    )
    try:
        rewritten = (prompt | llm | StrOutputParser()).invoke(
            {"history": _history_text(history), "question": user_question}
        ).strip()
    except Exception as exc:
        print(f"[WARNING] Could not rewrite the follow-up question ({exc})")
        return user_question
    print(f"[PROCESS] Follow-up rewritten as: '{rewritten}'")
    return rewritten or user_question


def prepare_query(user_question, search_query=None, calendar_name=None, history=None):
    """Retrieve the fragments for a question; search_query and calendar_name only steer retrieval."""
    if not os.path.exists(os.path.join(_vector_store_path(), "index.faiss")):
        return RagRequest(user_question, error="Error: No knowledge index found. Please run 'python core/ingestion.py' first to process the university documents.")

    llm = _llm(FALLBACK_LLM_MODEL)
    if llm is None:
        return RagRequest(user_question, error="Error: GROQ_API_KEY is not configured. Set it in the Render environment variables (or in a local .env file) and restart the app.")

    question = standalone_question(user_question, history, llm)
    if question != user_question:
        search_query = question
    search_query = search_query or question
    calendar_name = calendar_name or calendar_name_in(question)

    print(f"[PROCESS] Searching document sources for: '{search_query}'")
    vector_db = load_vector_store(_vector_store_path(), default_cache_dir())
    found_documents = retrieve_documents(vector_db, search_query, calendar_name)
    if not found_documents:
        return RagRequest(question, error="No sections within the uploaded documents match your query.")

    request = RagRequest(question)
    for number, doc in enumerate(found_documents, start=1):
        file_name = _file_name(doc.metadata.get("source", ""))
        page_number = doc.metadata.get("page", 0) + 1
        request.fragment_sources.append(f"PDF: {file_name} (Page {page_number})")
        request.context_text += f"[Fragment {number}] (Source: {file_name}, page {page_number}):\n{doc.page_content}\n\n"
    return request


def _answer_chain(model=LLM_MODEL):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Write your entire answer in {language}.\n\nOfficial document fragments:\n\n{context}\n\nUser question: {question}",
            ),
        ]
    )
    return prompt | _llm(model) | StrOutputParser()


def _chain_input(request):
    return {"context": request.context_text, "question": request.question, "language": ANSWER_LANGUAGE}


def _ai_error(exc):
    """Explain a Groq failure in plain words; a rate limit (HTTP 429) only needs a short wait."""
    if "429" in str(exc) or "rate limit" in str(exc).lower():
        limit = "daily limit, tokens per day" if "tokens per day" in str(exc) else "limit per minute"
        return (
            f"Error: the AI service (Groq) has reached its usage limit for now (HTTP 429, {limit}). "
            "Please try again in a few minutes."
        )
    return f"Error: the AI service could not answer ({exc})"


def stream_answer(request):
    """Yield the answer text piece by piece so the app can show it while it is written.

    When the main model is out of daily tokens before writing anything, the
    fallback model writes the answer instead.
    """
    for model in (LLM_MODEL, FALLBACK_LLM_MODEL):
        started = False
        try:
            for chunk in _answer_chain(model).stream(_chain_input(request)):
                started = True
                yield chunk
            return
        except Exception as exc:
            if started or model == FALLBACK_LLM_MODEL or not _is_daily_limit(exc):
                yield f"\n\n{_ai_error(exc)}"
                return
            print(f"[WARNING] {model} reached its daily limit; answering with {FALLBACK_LLM_MODEL}")


def normalize_citations(answer):
    """Write every citation as [n]: the model sometimes uses 【n】 or 【n†fragment n】."""
    return CITATION_PATTERN.sub(lambda match: f"[{match.group(1)}]", answer)


def cited_sources(answer, fragment_sources):
    """Return the sources of the fragments the answer cites, numbered like the citations.

    Falls back to the first retrieved fragments when the answer cites nothing,
    and returns nothing when the answer says the information was not found.
    """
    if answer.startswith("Error:") or "Error: the AI service" in answer:
        return []
    numbers = []
    for match in CITATION_PATTERN.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= len(fragment_sources) and number not in numbers:
            numbers.append(number)
    if not numbers:
        if any(marker in answer.lower() for marker in NOT_FOUND_MARKERS):
            return []
        numbers = list(range(1, min(FALLBACK_SOURCES, len(fragment_sources)) + 1))
    return [f"[{number}] {fragment_sources[number - 1]}" for number in sorted(numbers)]


def query_rag_system(user_question, search_query=None, calendar_name=None, history=None):
    """Answer a question in one call; returns (answer, cited sources)."""
    request = prepare_query(user_question, search_query, calendar_name, history)
    if request.error:
        return request.error, []
    try:
        answer = _answer_chain().invoke(_chain_input(request))
    except Exception as exc:
        if not _is_daily_limit(exc):
            return _ai_error(exc), []
        try:
            answer = _answer_chain(FALLBACK_LLM_MODEL).invoke(_chain_input(request))
        except Exception as fallback_exc:
            return _ai_error(fallback_exc), []
    if not answer.strip():
        return "Error: the AI service returned an empty answer.", []
    answer = normalize_citations(answer)
    return answer, cited_sources(answer, request.fragment_sources)


if __name__ == "__main__":
    test_response, test_sources = query_rag_system("What are the requirements for an academic procedure?")
    print(f"\nResponse: {test_response}")
    print(f"Sources: {test_sources}")
