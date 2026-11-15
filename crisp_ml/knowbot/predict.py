"""Phase 5: re-rank KnowBot's candidate fragments for a new question with the saved model."""

import json
import os
import time
from functools import lru_cache

import joblib

from .features import FEATURES, candidate_rows, knowbot_baseline_order
from .retrieval import load_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "reranker.joblib")
CARD_PATH = os.path.join(ROOT, "models", "model_card.json")
MAX_QUESTION_CHARS = 500


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def model_card():
    with open(CARD_PATH, encoding="utf-8") as card_file:
        return json.load(card_file)


def rerank(question, top=5):
    """Return the best fragments for the question, by the model and by KnowBot's current order.

    Raises ValueError when the question is empty or too long.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Write a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"The question is longer than {MAX_QUESTION_CHARS} characters.")

    started = time.perf_counter()
    frame = candidate_rows(load_corpus(), question)
    frame["probability"] = load_model().predict_proba(frame[FEATURES])[:, 1]
    model_order = frame.sort_values("probability", ascending=False).head(top)
    baseline_order = frame.loc[knowbot_baseline_order(frame)].head(top)

    def describe(rows):
        return [
            {
                "rank": rank,
                "document": row.document.replace(".pdf", ""),
                "page": int(row.page),
                "probability": round(float(row.probability), 4),
                "cosine": round(float(row.cosine), 4),
                "bm25": round(float(row.bm25), 3),
                "excerpt": " ".join(row.text.split())[:320],
                "text": " ".join(row.text.split()),
            }
            for rank, row in enumerate(rows.itertuples(), start=1)
        ]

    return {
        "question": question,
        "candidates": int(len(frame)),
        "reranked": describe(model_order),
        # The written answer uses the ten fragments ranked first, as KnowBot sends up to ten.
        "answer_fragments": describe(frame.sort_values("probability", ascending=False).head(10)),
        "knowbot_order": describe(baseline_order),
        "top_probability": round(float(model_order.probability.iloc[0]), 4),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "model": model_card()["model"],
    }
