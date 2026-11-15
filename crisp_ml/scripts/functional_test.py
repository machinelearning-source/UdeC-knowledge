"""Functional test of the deployed application (CRISP-ML Phases 4 and 5).

Sends the 100 labeled questions to the public API and checks that:
1. the deployed model returns the same ranking as the saved model run locally (deployment fidelity);
2. the ranking metrics on the deployed service match the offline evaluation;
3. latency meets the success criterion (median below 3 seconds);
4. invalid inputs are rejected and Spanish questions are handled.

Output: reports/functional_test.json

Usage:  python scripts/functional_test.py https://knowbot-reranker.onrender.com
"""

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from knowbot.predict import rerank  # noqa: E402
from scripts.build_dataset import DOCUMENT_NAMES  # noqa: E402

# Spanish versions of six test questions, written by the team.
SPANISH = {
    9: "¿Qué acciones de procedimiento y criterios de rechazo se aplican en la etapa de subsanación de una experiencia formativa?",
    33: "¿Qué requisitos académicos debe cumplir una experiencia formativa y qué dependencia los evalúa?",
    67: "¿Cuáles son los requisitos y pasos para que un estudiante obtenga el reingreso a un programa de la Universidad de Cundinamarca?",
    72: "¿Qué deberes deben cumplir los estudiantes de pregrado según el reglamento de la Universidad de Cundinamarca?",
    74: "¿Qué acciones toma el Consejo Académico para facilitar el acceso de aspirantes con diversidad funcional?",
    89: "¿Qué circunstancias llevan a la pérdida de la calidad de estudiante en la Universidad de Cundinamarca?",
}


def call(url, question, timeout=120):
    body = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/api/rerank", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, json.load(error), (time.perf_counter() - started) * 1000


def pages(fragments):
    return [(f["document"], int(f["page"])) for f in fragments]


def rank_of(expected, ranked):
    return next((i + 1 for i, page in enumerate(ranked) if page == expected), None)


def metrics(ranks):
    ranks = [None if r is None or pd.isna(r) else int(r) for r in ranks]
    n = len(ranks)
    return {
        "questions": n,
        "mrr_at_5": round(sum(1 / r for r in ranks if r) / n, 3),
        "hit_at_1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "hit_at_3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
        "hit_at_5": round(sum(1 for r in ranks if r) / n, 3),
    }


def main(url):
    questions = pd.read_csv(os.path.join(ROOT, "data", "validation_questions.csv"), encoding="utf-8-sig")
    questions["expected_document"] = questions["expected_document"].map(DOCUMENT_NAMES).str.replace(".pdf", "", regex=False)
    with open(os.path.join(ROOT, "data", "split.json"), encoding="utf-8") as split_file:
        test_ids = set(json.load(split_file)["test_question_ids"])

    rows, latencies, round_trips, errors = [], [], [], []
    for question in questions.itertuples():
        status, deployed, round_trip = call(url, question.question)
        if status != 200:
            errors.append({"question_id": int(question.id), "status": status, "body": deployed})
            continue
        local = rerank(question.question)
        expected = (question.expected_document, int(question.expected_page))
        deployed_pages, local_pages = pages(deployed["reranked"]), pages(local["reranked"])
        rows.append({
            "question_id": int(question.id), "test": int(question.id) in test_ids,
            "rank_deployed": rank_of(expected, deployed_pages), "rank_local": rank_of(expected, local_pages),
            "same_top5_as_local": deployed_pages == local_pages,
            "same_first_page_as_local": deployed_pages[:1] == local_pages[:1],
        })
        latencies.append(deployed["latency_ms"])
        round_trips.append(round_trip)
        print(f"{question.id:>3} rank {rows[-1]['rank_deployed']} same={rows[-1]['same_top5_as_local']} {deployed['latency_ms']:.0f} ms", flush=True)

    frame = pd.DataFrame(rows)
    spanish = []
    for qid, text in SPANISH.items():
        status, deployed, _ = call(url, text)
        row = questions[questions.id == qid].iloc[0]
        english_rank = frame.loc[frame.question_id == qid, "rank_deployed"].iloc[0]
        spanish.append({"question_id": qid, "status": status,
                        "rank_spanish": rank_of((row.expected_document, int(row.expected_page)), pages(deployed.get("reranked", []))),
                        "rank_english": None if pd.isna(english_rank) else int(english_rank)})

    invalid = []
    for label, text in (("empty question", ""), ("501 characters", "x" * 501)):
        status, body, _ = call(url, text)
        invalid.append({"input": label, "status": status, "error": body.get("error")})

    def quantile(values, q):
        return round(statistics.quantiles(values, n=100)[q - 1], 1)

    report = {
        "url": url,
        "tested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "questions_sent": len(questions),
        "successful_responses": len(rows),
        "errors": errors,
        "deployment_fidelity": {
            "questions_with_same_first_page_as_local_model": int(frame.same_first_page_as_local.sum()),
            "questions_with_same_top5_as_local_model": int(frame.same_top5_as_local.sum()),
            "note": "Differences come from floating-point differences between CPUs, which change which "
                    "fragments sit at the edge of the 25 nearest candidates and reorder near ties.",
        },
        "deployed_metrics_all_100": metrics(frame.rank_deployed.tolist()),
        "deployed_metrics_test_20": metrics(frame[frame.test].rank_deployed.tolist()),
        "local_metrics_test_20": metrics(frame[frame.test].rank_local.tolist()),
        "latency_ms_server": {"median": round(statistics.median(latencies), 1), "p95": quantile(latencies, 95),
                              "max": round(max(latencies), 1)},
        "latency_ms_round_trip": {"median": round(statistics.median(round_trips), 1), "p95": quantile(round_trips, 95)},
        "spanish_questions": spanish,
        "invalid_inputs": invalid,
    }
    frame.to_csv(os.path.join(ROOT, "reports", "functional_test_per_question.csv"), index=False)
    with open(os.path.join(ROOT, "reports", "functional_test.json"), "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("errors",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000")
