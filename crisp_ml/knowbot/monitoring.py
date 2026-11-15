"""Phase 6: prediction log, user feedback and monitoring indicators.

Every prediction is appended to logs/predictions.csv. The "source" column says
where it came from: "demo" (the web form), "api", or "simulated" (the test
questions replayed by the monitoring page). Simulated rows are operational test
data, not real users, and are labeled as such everywhere they are shown.
"""

import csv
import math
import os
import threading
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.environ.get("KNOWBOT_LOG_DIR", os.path.join(ROOT, "logs"))
PREDICTIONS = os.path.join(LOG_DIR, "predictions.csv")
FEEDBACK = os.path.join(LOG_DIR, "feedback.csv")
PREDICTION_FIELDS = ["timestamp", "source", "question_chars", "candidates", "top_probability", "top_document",
                     "top_page", "agrees_with_knowbot_order", "latency_ms", "error"]
FEEDBACK_FIELDS = ["timestamp", "question", "useful", "comment"]

# Thresholds of the monitoring plan (see the Monitoring and Maintenance phase).
LOW_CONFIDENCE = 0.30
MAX_LOW_CONFIDENCE_SHARE = 0.40
MAX_MEDIAN_LATENCY_MS = 3000
MAX_DOCUMENT_PSI = 0.25
MIN_USEFUL_SHARE = 0.70

# Share of each document among the expected answers of the training questions.
TRAINING_DOCUMENT_SHARE = {
    "Undergraduate Student Regulations": 0.45,
    "Agreement 005": 0.41,
    "Commission Report CA": 0.11,
    "Academic Calendar 2026": 0.03,
}

_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path, fields, row):
    os.makedirs(LOG_DIR, exist_ok=True)
    with _lock:
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow(row)


def log_prediction(question, result, source, error=""):
    row = {"timestamp": _now(), "source": source, "question_chars": len(question or ""), "error": error}
    if result:
        top, baseline = result["reranked"][0], result["knowbot_order"][0]
        row.update({
            "candidates": result["candidates"],
            "top_probability": result["top_probability"],
            "top_document": top["document"],
            "top_page": top["page"],
            "agrees_with_knowbot_order": int((top["document"], top["page"]) == (baseline["document"], baseline["page"])),
            "latency_ms": result["latency_ms"],
        })
    _append(PREDICTIONS, PREDICTION_FIELDS, row)


def log_feedback(question, useful, comment):
    _append(FEEDBACK, FEEDBACK_FIELDS, {"timestamp": _now(), "question": (question or "")[:500],
                                        "useful": useful, "comment": (comment or "")[:500]})


def read_log(limit=50):
    if not os.path.exists(PREDICTIONS):
        return []
    return pd.read_csv(PREDICTIONS).tail(limit).iloc[::-1].fillna("").to_dict("records")


def population_stability_index(expected, observed):
    """PSI between two category distributions; above 0.25 is usually read as a significant shift."""
    total = 0.0
    for category, expected_share in expected.items():
        e = max(expected_share, 1e-4)
        o = max(observed.get(category, 0.0), 1e-4)
        total += (o - e) * math.log(o / e)
    return total


def indicators():
    """Monitoring indicators over the logged predictions, split by source."""
    if not os.path.exists(PREDICTIONS):
        return {"rows": 0}
    log = pd.read_csv(PREDICTIONS)
    ok = log[log.error.isna() | (log.error == "")]
    result = {
        "rows": int(len(log)),
        "by_source": log.source.value_counts().to_dict(),
        "errors": int(len(log) - len(ok)),
    }
    if len(ok):
        shares = ok.top_document.value_counts(normalize=True).to_dict()
        result.update({
            "median_latency_ms": float(ok.latency_ms.median()),
            "p95_latency_ms": float(ok.latency_ms.quantile(0.95)),
            "mean_top_probability": float(ok.top_probability.mean()),
            "low_confidence_share": float((ok.top_probability < LOW_CONFIDENCE).mean()),
            "agreement_with_knowbot_order": float(ok.agrees_with_knowbot_order.mean()),
            "top_document_share": {k: round(v, 3) for k, v in shares.items()},
            "document_psi": population_stability_index(TRAINING_DOCUMENT_SHARE, shares),
        })
        result["alerts"] = [
            message for condition, message in (
                (result["low_confidence_share"] > MAX_LOW_CONFIDENCE_SHARE,
                 f"More than {MAX_LOW_CONFIDENCE_SHARE:.0%} of predictions have a top probability below {LOW_CONFIDENCE}."),
                (result["median_latency_ms"] > MAX_MEDIAN_LATENCY_MS,
                 f"Median latency above {MAX_MEDIAN_LATENCY_MS} ms."),
                (result["document_psi"] > MAX_DOCUMENT_PSI,
                 f"Document distribution shifted (PSI above {MAX_DOCUMENT_PSI})."),
            ) if condition
        ]
    if os.path.exists(FEEDBACK):
        feedback = pd.read_csv(FEEDBACK)
        if len(feedback):
            result["feedback_rows"] = int(len(feedback))
            result["useful_share"] = float((feedback.useful == "yes").mean())
    return result
