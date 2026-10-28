"""Evaluate KnowBot on the 100-question validation set.

Two kinds of evaluation:

1. Retrieval (free, no API calls): for every question, does the search find
   the expected document and page?
2. Answers (uses the Groq API): KnowBot answers a sample of questions and an
   LLM judge compares each answer with the text of the expected page.

Usage (from the project folder, with the virtual environment active):

    python eval/evaluate.py                  # retrieval only, all 100 questions
    python eval/evaluate.py --answers 20     # retrieval + 20 judged answers

Each judged answer uses two Groq calls (about 5,000 tokens). The Groq free tier
allows 200,000 tokens per day, so 20 answers use about half of the daily quota
and the app shares that quota. Results are saved after every question.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core import default_cache_dir
from core.calendar_names import calendar_name_in
from core.pipeline import (
    NOT_FOUND_MARKERS,
    _file_name,
    _llm,
    _vector_store_path,
    load_vector_store,
    query_rag_system,
    retrieve_documents,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(EVAL_DIR, "validation_questions.csv")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# The validation set was written when the PDFs still had their Spanish file names.
DOCUMENT_NAMES = {
    "Reglamento estudiantil Pregrado.pdf": "Undergraduate Student Regulations.pdf",
    "ACUERDO_N_005.pdf": "Agreement 005.pdf",
    "INFORME_COMISION_CA.pdf": "Commission Report CA.pdf",
    "CALENDARIO-ACADEMICO-2026-V19-2.pdf": "Academic Calendar 2026.pdf",
}

TOP_K = 6
PAGE_TOLERANCE = 1
RETRIES = 4
RETRY_WAIT_SECONDS = 30
SAMPLE_SEED = 7
DAILY_LIMIT_TEXT = "tokens per day"

JUDGE_PROMPT = """You grade the answers of a university regulations chatbot.
You receive a question, the reference text of the official page that answers it, and the chatbot answer.
The chatbot also reads other pages of the same documents, so extra details that are not in the reference are NOT errors unless they contradict it.
Reply with JSON only: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one short sentence>"}}
- correct: the answer gives the key information the question asks for, as stated in the reference, and nothing contradicts the reference.
- partial: the answer is right but leaves out an important part of what the question asks for.
- incorrect: the answer is wrong, contradicts the reference, or says the information was not found although the reference contains it."""


def load_questions():
    with open(QUESTIONS_PATH, encoding="utf-8-sig") as questions_file:
        rows = list(csv.DictReader(questions_file))
    for row in rows:
        row["expected_document"] = DOCUMENT_NAMES.get(row["expected_document"], row["expected_document"])
        row["expected_page"] = int(row["expected_page"])
    return rows


def _hit(retrieved, document, page, tolerance=0):
    return any(name == document and abs(number - page) <= tolerance for name, number in retrieved)


def _with_retries(action):
    for attempt in range(1, RETRIES + 1):
        try:
            return action()
        except Exception as exc:
            if attempt == RETRIES:
                raise
            print(f"   [retry {attempt}] {exc.__class__.__name__}: waiting {RETRY_WAIT_SECONDS}s")
            time.sleep(RETRY_WAIT_SECONDS)


def _answer_with_retries(question):
    """query_rag_system reports API errors as text, so rate limits (HTTP 429) are retried here."""
    for attempt in range(1, RETRIES + 1):
        answer, sources = query_rag_system(question)
        if not ("429" in answer or "rate limit" in answer.lower()) or attempt == RETRIES:
            return answer, sources
        print(f"   [retry {attempt}] rate limit: waiting {RETRY_WAIT_SECONDS}s")
        time.sleep(RETRY_WAIT_SECONDS)


def _is_not_found(answer):
    return any(marker in answer.lower() for marker in NOT_FOUND_MARKERS)


def evaluate_retrieval(rows):
    vector_db = load_vector_store(_vector_store_path(), default_cache_dir())
    results = []
    for row in rows:
        question = row["question"]
        documents = retrieve_documents(vector_db, question, calendar_name_in(question))
        retrieved = [
            (_file_name(doc.metadata.get("source", "")), doc.metadata.get("page", 0) + 1) for doc in documents
        ]
        top = retrieved[:TOP_K]
        document, page = row["expected_document"], row["expected_page"]
        results.append(
            {
                "id": row["id"],
                "question": question,
                "expected": f"{document} p.{page}",
                "top1": f"{top[0][0]} p.{top[0][1]}" if top else "",
                "doc_top1": bool(top) and top[0][0] == document,
                "page_top1": bool(top) and top[0] == (document, page),
                "doc_topk": any(name == document for name, _ in top),
                "page_topk": _hit(top, document, page),
                "page_topk_tolerant": _hit(top, document, page, PAGE_TOLERANCE),
                "page_all": _hit(retrieved, document, page),
            }
        )
    return results


def _reference_text(vector_db, document, page):
    chunks = [
        doc.page_content
        for doc in vector_db.docstore._dict.values()
        if _file_name(doc.metadata.get("source", "")) == document and doc.metadata.get("page", 0) + 1 == page
    ]
    return "\n".join(chunks)[:6000]


def _judge(question, reference, answer):
    prompt = ChatPromptTemplate.from_messages(
        [("system", JUDGE_PROMPT), ("human", "Question: {question}\n\nReference text:\n{reference}\n\nChatbot answer:\n{answer}")]
    )
    raw = _with_retries(
        lambda: (prompt | _llm() | StrOutputParser()).invoke(
            {"question": question, "reference": reference, "answer": answer}
        )
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    return verdict.get("verdict", "unparsed"), verdict.get("reason", raw[:200])


def evaluate_answers(rows, sample_size, delay):
    vector_db = load_vector_store(_vector_store_path(), default_cache_dir())
    sample = random.Random(SAMPLE_SEED).sample(rows, min(sample_size, len(rows)))
    results = []
    for position, row in enumerate(sample, start=1):
        print(f"[{position}/{len(sample)}] {row['question'][:80]}")
        answer, sources = _answer_with_retries(row["question"])
        if answer.startswith("Error:"):
            print(f"   {answer[:160]}")
            if DAILY_LIMIT_TEXT in answer:
                print("[STOP] The Groq daily token limit is reached; saving the questions graded so far.")
                break
        reference = _reference_text(vector_db, row["expected_document"], row["expected_page"])
        try:
            verdict, reason = _judge(row["question"], reference, answer)
        except Exception as exc:
            if DAILY_LIMIT_TEXT in str(exc):
                print("[STOP] The Groq daily token limit is reached; saving the questions graded so far.")
                break
            verdict, reason = "unjudged", f"judge error: {exc.__class__.__name__}"
        expected = f"PDF: {row['expected_document']} (Page {row['expected_page']})"
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "verdict": verdict,
                "reason": reason,
                "cites_expected_page": any(source.endswith(expected) for source in sources),
                "says_not_found": _is_not_found(answer),
                "api_error": answer.startswith("Error:"),
                "answer": answer,
                "sources": " | ".join(sources),
            }
        )
        _write_csv(os.path.join(RESULTS_DIR, "answers.csv"), results)
        time.sleep(delay)
    return results


def _rate(results, key):
    return 100 * sum(1 for result in results if result[key]) / len(results) if results else 0.0


def _write_csv(path, results):
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def _baseline(rows):
    """Results stored in the validation set by the first version (MMR only, k=6)."""
    with open(QUESTIONS_PATH, encoding="utf-8-sig") as questions_file:
        original = list(csv.DictReader(questions_file))
    total = len(original)
    return {
        "doc_top1": 100 * sum(r["top1_document"] == r["expected_document"] for r in original) / total,
        "page_top1": 100 * sum(
            r["top1_document"] == r["expected_document"] and r["top1_page"] == r["expected_page"] for r in original
        ) / total,
        "doc_topk": 100 * sum(r["matched_document"] == "yes" for r in original) / total,
        "page_topk_tolerant": 100 * sum(r["matched_page"] == "yes" for r in original) / total,
    }


def write_summary(rows, retrieval, answers):
    baseline = _baseline(rows)
    lines = [
        "# KnowBot evaluation results",
        "",
        f"Generated on {datetime.now():%Y-%m-%d %H:%M} with `python eval/evaluate.py`.",
        f"Validation set: {len(rows)} questions (`eval/validation_questions.csv`).",
        "",
        "## Retrieval",
        "",
        f"A hit means the expected document (and page) is among the first {TOP_K} retrieved fragments.",
        f"'Page ±{PAGE_TOLERANCE}' also accepts the neighbouring page, because a fragment can start on one page and end on the next.",
        "",
        "| Metric | First version (MMR, k=6) | Current version |",
        "|---|---|---|",
        f"| Correct document, first result | {baseline['doc_top1']:.0f}% | {_rate(retrieval, 'doc_top1'):.0f}% |",
        f"| Correct page, first result | {baseline['page_top1']:.0f}% | {_rate(retrieval, 'page_top1'):.0f}% |",
        f"| Correct document in top {TOP_K} | {baseline['doc_topk']:.0f}% | {_rate(retrieval, 'doc_topk'):.0f}% |",
        f"| Correct page in top {TOP_K} | — | {_rate(retrieval, 'page_topk'):.0f}% |",
        f"| Correct page ±{PAGE_TOLERANCE} in top {TOP_K} | {baseline['page_topk_tolerant']:.0f}% | {_rate(retrieval, 'page_topk_tolerant'):.0f}% |",
        f"| Correct page in all fragments sent to the model (up to 10) | — | {_rate(retrieval, 'page_all'):.0f}% |",
        "",
        "The first-version page column comes from the original validation run, whose exact page rule is not documented; treat that comparison as approximate.",
        "",
    ]
    if answers:
        total = len(answers)
        counts = {verdict: sum(1 for a in answers if a["verdict"] == verdict) for verdict in ("correct", "partial", "incorrect")}
        lines += [
            "## Answers",
            "",
            f"{total} questions sampled at random (seed {SAMPLE_SEED}), answered by KnowBot and graded by an LLM judge against the text of the expected page.",
            "",
            "| Verdict | Questions | Share |",
            "|---|---|---|",
        ]
        for verdict, count in counts.items():
            lines.append(f"| {verdict.capitalize()} | {count} | {100 * count / total:.0f}% |")
        other = total - sum(counts.values())
        if other:
            lines.append(f"| Not graded | {other} | {100 * other / total:.0f}% |")
        lines += [
            "",
            f"Correct or partial (useful answers): {100 * (counts['correct'] + counts['partial']) / total:.0f}%.",
            f"The answer cites the expected page in {_rate(answers, 'cites_expected_page'):.0f}% of the sampled questions.",
            f"The answer says the information was not found in {_rate(answers, 'says_not_found'):.0f}% of the sampled questions "
            "(every question of the set has an answer in the documents, so these are retrieval misses).",
            f"API errors after retries: {sum(1 for a in answers if a['api_error'])}.",
            "",
            "An LLM judge is a fast approximation of a human review; the per-question reasons are in `answers.csv`.",
            "",
        ]
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=int, default=0, help="number of questions to answer and grade (uses the Groq API)")
    parser.add_argument(
        "--delay", type=float, default=20.0,
        help="seconds to wait between answered questions (the Groq free tier allows 8000 tokens per minute)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = load_questions()

    print(f"[PROCESS] Evaluating retrieval on {len(rows)} questions...")
    retrieval = evaluate_retrieval(rows)
    _write_csv(os.path.join(RESULTS_DIR, "retrieval.csv"), retrieval)

    answers = []
    if args.answers:
        print(f"[PROCESS] Answering and grading {args.answers} questions...")
        answers = evaluate_answers(rows, args.answers, args.delay)
        _write_csv(os.path.join(RESULTS_DIR, "answers.csv"), answers)

    print()
    print(write_summary(rows, retrieval, answers))


if __name__ == "__main__":
    main()
