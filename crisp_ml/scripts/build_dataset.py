"""Phase 2 (data engineering): build the re-ranking dataset from the validation questions.

Input:  data/validation_questions.csv (100 questions labeled with the document and page that answer them)
Output: data/raw_candidates.csv    one row per (question, candidate fragment), before cleaning
        data/dataset.csv           cleaned rows used for modeling
        data/split.json            question ids of the training/CV and test partitions
        reports/data_quality.json  counts used in the report and the app

Usage:  python scripts/build_dataset.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from knowbot.features import candidate_rows  # noqa: E402
from knowbot.retrieval import load_corpus  # noqa: E402

DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
SEED = 42
TEST_QUESTIONS = 20

# The validation set was written when the PDFs still had their Spanish file names.
DOCUMENT_NAMES = {
    "Reglamento estudiantil Pregrado.pdf": "Undergraduate Student Regulations.pdf",
    "ACUERDO_N_005.pdf": "Agreement 005.pdf",
    "INFORME_COMISION_CA.pdf": "Commission Report CA.pdf",
    "CALENDARIO-ACADEMICO-2026-V19-2.pdf": "Academic Calendar 2026.pdf",
}


def load_questions():
    questions = pd.read_csv(os.path.join(DATA, "validation_questions.csv"), encoding="utf-8-sig")
    questions = questions[["id", "question", "expected_document", "expected_page"]].copy()
    questions["expected_document"] = questions["expected_document"].map(DOCUMENT_NAMES)
    return questions


def stratified_question_split(questions):
    """Hold out 20% of the questions per document for the final test (whole questions, never rows)."""
    rng = np.random.default_rng(SEED)
    test_ids = []
    for _, group in questions.groupby("expected_document"):
        ids = group["id"].tolist()
        rng.shuffle(ids)
        test_ids += ids[: max(1, round(len(ids) * TEST_QUESTIONS / len(questions)))]
    test_ids = sorted(test_ids)
    train_ids = sorted(set(questions["id"]) - set(test_ids))
    return train_ids, test_ids


def main():
    os.makedirs(REPORTS, exist_ok=True)
    corpus = load_corpus()
    questions = load_questions()

    quality = {
        "questions": int(len(questions)),
        "duplicate_questions": int(questions["question"].str.strip().str.lower().duplicated().sum()),
        "missing_question_fields": int(questions.isna().sum().sum()),
        "questions_per_document": questions["expected_document"].value_counts().to_dict(),
        "indexed_fragments": len(corpus.chunks),
        "duplicate_fragments": int(pd.Series([c.page_content for c in corpus.chunks]).duplicated().sum()),
    }
    pages = {(r.metadata["source"].split("/")[-1].split("\\")[-1], r.metadata.get("page", 0) + 1) for r in corpus.chunks}
    quality["questions_whose_page_is_not_indexed"] = [
        int(q.id) for q in questions.itertuples() if (q.expected_document, q.expected_page) not in pages
    ]

    frames = []
    for q in questions.itertuples():
        frame = candidate_rows(corpus, q.question)
        frame.insert(0, "question_id", q.id)
        frame["label"] = ((frame.document == q.expected_document) & (frame.page == q.expected_page)).astype(int)
        frame["label_near"] = ((frame.document == q.expected_document) & ((frame.page - q.expected_page).abs() <= 1)).astype(int)
        frames.append(frame)
        print(f"question {q.id:>3}: {len(frame)} candidates, {frame.label.sum()} on the expected page")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(os.path.join(DATA, "raw_candidates.csv"), index=False, encoding="utf-8")

    # Cleaning: the same fragment text can be indexed twice; keep one copy per question.
    dataset = raw.drop_duplicates(subset=["question_id", "text"]).copy()
    quality["raw_rows"] = int(len(raw))
    quality["duplicate_candidate_rows_removed"] = int(len(raw) - len(dataset))
    quality["rows"] = int(len(dataset))
    quality["positive_rows"] = int(dataset.label.sum())
    quality["positive_rate"] = round(float(dataset.label.mean()), 4)
    quality["missing_bm25_rank"] = int(dataset.bm25_rank.isna().sum())
    covered = dataset.groupby("question_id").label.max()
    quality["questions_with_expected_page_in_candidates"] = int(covered.sum())
    quality["questions_without_expected_page_in_candidates"] = covered[covered == 0].index.astype(int).tolist()
    dataset.drop(columns=["text"]).to_csv(os.path.join(DATA, "dataset.csv"), index=False, encoding="utf-8")

    train_ids, test_ids = stratified_question_split(questions)
    with open(os.path.join(DATA, "split.json"), "w", encoding="utf-8") as split_file:
        json.dump({"seed": SEED, "train_cv_question_ids": train_ids, "test_question_ids": test_ids}, split_file, indent=2)
    quality["train_cv_questions"] = len(train_ids)
    quality["test_questions"] = len(test_ids)
    with open(os.path.join(REPORTS, "data_quality.json"), "w", encoding="utf-8") as report:
        json.dump(quality, report, indent=2, ensure_ascii=False)
    print(json.dumps(quality, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
