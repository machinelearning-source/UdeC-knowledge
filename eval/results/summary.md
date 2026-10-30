# KnowBot evaluation results

Generated on 2026-10-30 11:42 with `python eval/evaluate.py`.
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
| Correct page ±1 in top 6 | 73% | 89% |
| Correct page in all fragments sent to the model (up to 10) | — | 87% |

The first-version page column comes from the original validation run, whose exact page rule is not documented; treat that comparison as approximate.

## Answers

20 questions sampled at random (seed 7), answered by KnowBot and graded by an LLM judge against the text of the expected page.

| Verdict | Questions | Share |
|---|---|---|
| Correct | 12 | 60% |
| Partial | 3 | 15% |
| Incorrect | 5 | 25% |

Correct or partial (useful answers): 75%.
The answer cites the expected page in 80% of the sampled questions.
The answer says the information was not found in 15% of the sampled questions (every question of the set has an answer in the documents, so these are retrieval misses).
API errors after retries: 0.

An LLM judge is a fast approximation of a human review; the per-question reasons are in `answers.csv`.
