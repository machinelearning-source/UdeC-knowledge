<p align="center">
  <img src="static/knowbot-logo.png" alt="KnowBot - Ask. Learn. Know." width="380">
</p>

# KnowBot Re-ranker: a CRISP-ML project

A Machine Learning model that re-ranks the fragments retrieved by [KnowBot](https://udec-knowledge.streamlit.app/), the chatbot that answers questions about the official documents of the University of Cundinamarca. The whole project follows the six CRISP-ML phases and is documented in a Flask application that also serves the model.

- **Application (six phases and model demo): <https://knowbot-reranker.onrender.com>**
- KnowBot chatbot: <https://udec-knowledge.streamlit.app/>
- KnowBot repository: <https://github.com/machinelearning-source/UdeC-knowledge>

## The problem in one paragraph

KnowBot retrieves up to ten fragments of four official PDFs and a language model writes the answer from them. When the page that answers the question is not among the first fragments, the answer is incomplete or says "not found". This project learns to score each candidate fragment (probability that it comes from the page that answers the question) and sorts the candidates by that score. The learning task is supervised binary classification used for ranking.

## Results

Grouped 5-fold cross-validation by question on 80 training questions (best setting of each model), selection metric MRR:

| Ranking | MRR | Hit@1 | Hit@3 | Page ±1 at rank 1 |
|---|---|---|---|---|
| Vector search only | 0.544 | 41.2% | 62.5% | 57.5% |
| KnowBot order (baseline) | 0.585 | 43.8% | 67.5% | 57.5% |
| Logistic Regression, C = 0.1 | 0.590 | 50.0% | 61.2% | 58.8% |
| **Random Forest**, 300 trees, min 5 samples per leaf (selected) | **0.594** | **50.0%** | 62.5% | **66.2%** |
| Gradient Boosting, depth 2 | 0.568 | 46.2% | 58.8% | 61.2% |

Independent test set (20 questions held out from the start, used once):

| Ranking | MRR | Hit@1 | Hit@3 | Page ±1 at rank 1 |
|---|---|---|---|---|
| Vector search only | 0.676 | 55% | 75% | 60% |
| KnowBot order (baseline) | 0.724 | 60% | 75% | 65% |
| Selected model | 0.658 | 55% | 75% | 60% |

The model beats KnowBot's order in cross-validation, but by less than one standard deviation, and it does not beat it on the test set. The gain is therefore not confirmed. The model is deployed in shadow mode: it runs and is logged next to KnowBot's order, without replacing it, until more labeled questions are available. All figures are produced by the scripts below and shown in the application.

## Project structure

```
crisp_ml/
├── app.py                      # Flask app: overview, six phases, demo, API, monitoring
├── knowbot/
│   ├── retrieval.py            # FAISS index, BM25 and exact lookups (from KnowBot)
│   ├── embeddings.py           # FastEmbed multilingual embeddings (from KnowBot), slim mode for 512 MB
│   ├── calendar_names.py       # Official names of the calendar activities (from KnowBot)
│   ├── features.py             # Candidate generation and the 23 features (training and serving)
│   ├── model.py                # Preprocessing pipeline, models, grid and ranking metrics
│   ├── predict.py              # Loads the saved model and re-ranks a new question
│   └── monitoring.py           # Prediction and feedback logs, monitoring indicators
├── scripts/
│   ├── build_dataset.py        # Phase 2: dataset, cleaning, question-level split
│   └── train.py                # Phases 3-4: cross-validation, tuning, selection, test
├── data/
│   ├── validation_questions.csv  # 100 labeled questions (expected document and page)
│   ├── raw_candidates.csv        # Dataset before cleaning (with fragment text)
│   ├── dataset.csv               # Dataset used for modeling
│   └── split.json                # Training/CV and test question ids
├── models/
│   ├── reranker.joblib         # Selected model with its fitted preprocessing
│   └── model_card.json         # Configuration, metrics, versions
├── reports/                    # Metrics (CSV/JSON), charts (PNG), experiment log
├── vector_store/               # KnowBot FAISS index (290 fragments)
├── templates/, static/         # Web pages and style
├── requirements.txt
└── render.yaml                 # Deployment on Render
```

## Reproduce the project

Requirements: Python 3.12. No API key is needed.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python scripts/build_dataset.py # Phase 2: data/dataset.csv, data/split.json, reports/data_quality.json
python scripts/train.py         # Phases 3-4: reports/*, models/reranker.joblib, models/model_card.json
python app.py                   # http://127.0.0.1:5000
```

The first run downloads the embedding model (about 240 MB) to a local cache. Results are deterministic (seed 42). The trained model and all reports are already in the repository, so `python app.py` works without retraining.

## Use the model

- Web: the **Try the model** page.
- API:

```bash
curl -X POST http://127.0.0.1:5000/api/rerank -H "Content-Type: application/json" \
     -d '{"question": "What is the minimum number of students required to open an academic group?"}'
```

The response contains the five best fragments by the model (`reranked`) and by KnowBot's order (`knowbot_order`), the number of candidates, the top probability and the latency. Questions must have 1 to 500 characters.

## Deployment

`render.yaml` deploys the app on Render as a free web service (512 MB of RAM, 0.1 CPU): the build installs the dependencies and prepares the embedding model, and `gunicorn app:app` serves it. `/health` is the health check. Prediction and feedback logs are written to `logs/` (reset when the service restarts).

**Fitting in 512 MB.** Loaded as usual, the multilingual embedding model makes the app use about 740 MB. `knowbot/embeddings.py` has a slim mode that brings it to about 355 MB with identical vectors:

- the 250,002 x 384 vocabulary table (183 MB in float16, which onnxruntime expands to float32) is moved out of the ONNX graph into a memory-mapped file, and only the rows of each question's tokens are read;
- the tokenizer runs on the model's original SentencePiece file instead of `tokenizer.json`, which takes about 250 MB in memory; the token ids are the same.

Both changes were checked against the standard model: identical token ids on 400 texts and a maximum difference of 0.0 between the vectors of the 100 questions.

## Methodology summary

| Phase | Where |
|---|---|
| 1. Business and Data Understanding | App, phase 1; `reports/data_quality.json` |
| 2. Data Engineering | `scripts/build_dataset.py`, `knowbot/features.py`, `knowbot/model.py`; app, phase 2 |
| 3. Model Engineering | `scripts/train.py`, `reports/cv_summary.csv`, `reports/experiments/`; app, phase 3 |
| 4. Model Evaluation | `reports/cv_best.csv`, `reports/test_metrics.json`, `reports/*.png`; app, phase 4 |
| 5. Model Deployment | `models/`, `knowbot/predict.py`, `app.py`, `render.yaml`; app, phase 5 and demo |
| 6. Monitoring and Maintenance | `knowbot/monitoring.py`; app, phase 6 |

## Authors

Oscar Javier Bello Daza, Mariana León Triana, Luis Orlando Ramírez Mosquera, Luis Alberto Rebolledo Ariza. Systems and Computing Engineering, Universidad de Cundinamarca, Chía Extension.
