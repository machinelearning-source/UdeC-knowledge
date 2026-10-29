# KnowBot evaluation results

Generated on 2026-10-29 09:38 with `python eval/evaluate.py`.
Validation set: 100 questions (`eval/validation_questions.csv`).

## Retrieval

A hit means the expected document (and page) is among the first 6 retrieved fragments.
'Page ±1' also accepts the neighbouring page, because a fragment can start on one page and end on the next.

| Metric | First version (MMR, k=6) | Current version |
|---|---|---|
| Correct document, first result | 80% | 81% |
| Correct page, first result | 44% | 46% |
| Correct document in top 6 | 99% | 98% |
| Correct page in top 6 | — | 82% |
| Correct page ±1 in top 6 | 73% | 93% |
| Correct page in all fragments sent to the model (up to 10) | — | 84% |

The first-version page column comes from the original validation run, whose exact page rule is not documented; treat that comparison as approximate.

## Answers

Run on 2026-10-29 with the retrieval that existed before the keyword (BM25) search was added; it must be re-run to reflect the current retrieval.

20 questions sampled at random (seed 7), answered by KnowBot and graded by an LLM judge against the text of the expected page.

| Verdict | Questions | Share |
|---|---|---|
| Correct | 10 | 50% |
| Partial | 2 | 10% |
| Incorrect | 8 | 40% |

Correct or partial (useful answers): 60%.
The answer cites the expected page in 60% of the sampled questions.
The answer says the information was not found in 20% of the sampled questions (every question of the set has an answer in the documents, so these are retrieval misses).
API errors after retries: 0.

An LLM judge is a fast approximation of a human review; the per-question reasons are in `answers.csv`.
