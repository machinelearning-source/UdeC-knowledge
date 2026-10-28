# KnowBot evaluation results

Generated on 2026-10-28 12:24 with `python eval/evaluate.py`.
Validation set: 100 questions (`eval/validation_questions.csv`).

## Retrieval

A hit means the expected document (and page) is among the first 6 retrieved fragments.
'Page ±1' also accepts the neighbouring page, because a fragment can start on one page and end on the next.

| Metric | First version (MMR, k=6) | Current version |
|---|---|---|
| Correct document, first result | 80% | 81% |
| Correct page, first result | 44% | 46% |
| Correct document in top 6 | 99% | 99% |
| Correct page in top 6 | — | 76% |
| Correct page ±1 in top 6 | 73% | 88% |
| Correct page in all fragments sent to the model (up to 10) | — | 76% |

The first-version page column comes from the original validation run, whose exact page rule is not documented; treat that comparison as approximate.
