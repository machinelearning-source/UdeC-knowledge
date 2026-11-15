"""Preprocessing, candidate models and ranking metrics of the re-ranker."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Heavy-tailed counts (BM25 scores, ranks, fragment lengths) are log-scaled.
LOG_FEATURES = ["bm25", "bm25_rank", "cosine_rank", "chunk_chars", "knowbot_rank", "page_bm25_sum"]
OTHER_NUMERIC = [name for name in NUMERIC_FEATURES if name not in LOG_FEATURES]
# A fragment with no word in common with the question has no keyword rank: it is
# treated as ranked after every fragment of the corpus.
MISSING_RANK = 300


def make_preprocessor():
    """Fitted inside every training fold only, so validation and test data never leak into it."""
    log_numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_RANK)),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [
            ("log_numeric", log_numeric, LOG_FEATURES),
            ("numeric", numeric, OTHER_NUMERIC),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_model(name, params, seed=42):
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, **params)
    elif name == "random_forest":
        estimator = RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1, **params)
    elif name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **params)
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])


SEARCH_SPACE = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [
        {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf}
        for depth in (4, 8, None)
        for leaf in (1, 5)
    ],
    "gradient_boosting": [
        {"learning_rate": rate, "max_depth": depth, "max_iter": 200, "l2_regularization": 1.0}
        for rate in (0.03, 0.1)
        for depth in (2, 3, 5)
    ],
}


def ranking_metrics(frame, score_column):
    """Per-question ranking quality.

    For each question the fragments are sorted by the score; the metrics look at
    where the first fragment from the expected page lands. Questions whose
    expected page is not among the candidates count as misses.
    """
    hits1, hits3, hits5, near1, near3, reciprocal = [], [], [], [], [], []
    for _, group in frame.groupby("question_id"):
        ordered = group.sort_values(score_column, ascending=False)
        labels = ordered["label"].to_numpy()
        near = ordered["label_near"].to_numpy()
        positions = np.flatnonzero(labels)
        first = positions[0] + 1 if len(positions) else None
        hits1.append(bool(labels[:1].any()))
        hits3.append(bool(labels[:3].any()))
        hits5.append(bool(labels[:5].any()))
        near1.append(bool(near[:1].any()))
        near3.append(bool(near[:3].any()))
        reciprocal.append(1 / first if first else 0.0)
    return {
        "mrr": float(np.mean(reciprocal)),
        "hit_at_1": float(np.mean(hits1)),
        "hit_at_3": float(np.mean(hits3)),
        "hit_at_5": float(np.mean(hits5)),
        "near_hit_at_1": float(np.mean(near1)),
        "near_hit_at_3": float(np.mean(near3)),
    }


def order_scores(frame, order):
    """Turn a fragment order (an index) into a descending score, for ranking_metrics."""
    scores = pd.Series(0.0, index=frame.index)
    scores.loc[order] = np.arange(len(order), 0, -1, dtype=float)
    return scores
