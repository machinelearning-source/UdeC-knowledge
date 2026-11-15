"""Candidate fragments and features for the re-ranker.

For a question, the candidates are the fragments any KnowBot search would bring
back: the closest ones by meaning (vector search), the best keyword (BM25)
matches, and the exact calendar-activity and article matches. Each
(question, fragment) pair becomes one row with the features below; the model
predicts whether that fragment comes from the page that answers the question.
"""

import numpy as np
import pandas as pd

from .retrieval import (
    article_matches,
    calendar_activity_matches,
    file_name,
    is_calendar_chunk,
    keyword_tokens,
    looks_like_calendar_question,
)

VECTOR_POOL = 25
KEYWORD_POOL = 15

NUMERIC_FEATURES = [
    "cosine",
    "cosine_rank",
    "cosine_gap",
    "bm25",
    "bm25_rank",
    "bm25_relative",
    "word_overlap",
    "calendar_match",
    "article_match",
    "is_calendar_chunk",
    "calendar_question",
    "calendar_question_and_chunk",
    "chunk_chars",
    "page_position",
    "question_words",
    "knowbot_rank",
    "page_candidates",
    "page_max_cosine",
    "page_bm25_sum",
    "document_share",
    "cosine_z",
    "bm25_z",
    "page_max_cosine_z",
]
CATEGORICAL_FEATURES = ["document"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def candidate_rows(corpus, question):
    """Return one row per candidate fragment, with its features and identity."""
    query = corpus.query_vector(question)
    cosine = corpus.vectors @ query
    cosine_order = np.argsort(-cosine)
    cosine_rank = np.empty(len(cosine), dtype=int)
    cosine_rank[cosine_order] = np.arange(1, len(cosine) + 1)

    bm25 = corpus.keyword_index.scores(question)
    bm25_order = np.argsort(-bm25)
    bm25_rank = np.full(len(bm25), np.nan)
    matched = bm25 > 0
    bm25_rank[bm25_order] = np.arange(1, len(bm25) + 1)
    bm25_rank[~matched] = np.nan  # no shared word: the keyword rank is undefined (missing value)

    calendar = calendar_activity_matches(corpus.chunks, question)
    articles = article_matches(corpus.chunks, question)
    pool = list(cosine_order[:VECTOR_POOL]) + [i for i in bm25_order[:KEYWORD_POOL] if matched[i]]
    pool += sorted(calendar) + sorted(articles)

    question_words = set(keyword_tokens(question))
    calendar_question = int(looks_like_calendar_question(question) or bool(calendar))
    rows, seen = [], set()
    for index in pool:
        doc = corpus.chunks[index]
        if index in seen:
            continue
        seen.add(index)
        name = file_name(doc.metadata.get("source", ""))
        page = doc.metadata.get("page", 0) + 1
        chunk_words = set(keyword_tokens(doc.page_content))
        rows.append(
            {
                "chunk_index": int(index),
                "document": name,
                "page": page,
                "cosine": float(cosine[index]),
                "cosine_rank": int(cosine_rank[index]),
                "cosine_gap": float(cosine[cosine_order[0]] - cosine[index]),
                "bm25": float(bm25[index]),
                "bm25_rank": bm25_rank[index],
                "bm25_relative": float(bm25[index] / bm25.max()) if bm25.max() > 0 else 0.0,
                "word_overlap": len(question_words & chunk_words) / max(len(question_words), 1),
                "calendar_match": int(index in calendar),
                "article_match": int(index in articles),
                "is_calendar_chunk": int(is_calendar_chunk(doc)),
                "calendar_question": calendar_question,
                "calendar_question_and_chunk": int(calendar_question and is_calendar_chunk(doc)),
                "chunk_chars": len(doc.page_content),
                "page_position": page / corpus.max_page[name],
                "question_words": len(question.split()),
                "text": doc.page_content,
            }
        )
    frame = pd.DataFrame(rows)
    return add_context_features(frame)


def add_context_features(frame):
    """Features that compare a fragment with the other candidates of the same question.

    Domain knowledge: an answer page is usually hit by several fragments, and the
    right document tends to dominate the candidate list (the first result already
    has the right document in 81% of the validation questions).
    """
    order = knowbot_baseline_order(frame)
    frame["knowbot_rank"] = pd.Series(np.arange(1, len(order) + 1), index=order).loc[frame.index]
    page = frame.groupby(["document", "page"])
    frame["page_candidates"] = page["chunk_index"].transform("count")
    frame["page_max_cosine"] = page["cosine"].transform("max")
    frame["page_bm25_sum"] = page["bm25"].transform("sum")
    frame["document_share"] = frame.groupby("document")["chunk_index"].transform("count") / len(frame)
    # Scores are only comparable inside one question: an easy question has high
    # similarities everywhere. Standardizing per question removes that offset.
    for column in ("cosine", "bm25", "page_max_cosine"):
        spread = frame[column].std(ddof=0)
        frame[f"{column}_z"] = (frame[column] - frame[column].mean()) / spread if spread > 0 else 0.0
    return frame


def vector_baseline_order(frame):
    """Order of plain vector search (what FAISS alone returns)."""
    return frame.sort_values("cosine_rank").index


def knowbot_baseline_order(frame):
    """Order of the KnowBot app before the re-ranker, without the LLM translation step.

    Exact calendar and article matches first, then the best vector hit, the two
    best keyword hits, and the rest of the vector hits.
    """
    exact = frame[(frame.calendar_match == 1) | (frame.article_match == 1)].sort_values("cosine_rank")
    rest = frame.drop(exact.index).sort_values("cosine_rank")
    keyword = rest[rest.bm25_rank.notna()].sort_values("bm25_rank").head(2)
    order = list(exact.index) + list(rest.index[:1]) + list(keyword.index) + list(rest.index[1:])
    return pd.Index(list(dict.fromkeys(order)))
