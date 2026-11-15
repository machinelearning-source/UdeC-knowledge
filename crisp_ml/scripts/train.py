"""Phases 3 and 4: train, tune and compare the re-ranking models, then test the selected one.

1. Grouped 5-fold cross-validation on the 80 training questions (all candidates
   of a question stay in the same fold, so no question is seen in training and
   validation at the same time). Every hyperparameter setting of the three
   algorithms is scored by the mean reciprocal rank (MRR) of the expected page.
2. The setting with the best mean validation MRR is refitted on the 80
   questions and evaluated once on the 20 held-out test questions.

Outputs: models/reranker.joblib, models/model_card.json, reports/*.csv|json|png

Usage:  python scripts/train.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from knowbot.features import FEATURES, knowbot_baseline_order, vector_baseline_order  # noqa: E402
from knowbot.model import SEARCH_SPACE, make_model, order_scores, ranking_metrics  # noqa: E402
from scripts.build_dataset import DOCUMENT_NAMES  # noqa: E402

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
FOLDS = 5
SEED = 42
COLORS = {"vector_baseline": "#b9c2bb", "knowbot_baseline": "#6b7a70", "logistic_regression": "#5fa877",
          "random_forest": "#00482b", "gradient_boosting": "#e0b400"}


def baseline_scores(frame):
    vector = pd.concat([order_scores(g, vector_baseline_order(g)) for _, g in frame.groupby("question_id")])
    knowbot = pd.concat([order_scores(g, knowbot_baseline_order(g)) for _, g in frame.groupby("question_id")])
    return vector.loc[frame.index], knowbot.loc[frame.index]


def evaluate(frame, scores):
    scored = frame.assign(score=np.asarray(scores))
    metrics = ranking_metrics(scored, "score")
    metrics["roc_auc"] = float(roc_auc_score(frame.label, scores))
    metrics["pr_auc"] = float(average_precision_score(frame.label, scores))
    return metrics


def cross_validate(train):
    folds = list(GroupKFold(n_splits=FOLDS).split(train, train.label, groups=train.question_id))
    records = []
    for fold, (_, valid_index) in enumerate(folds, start=1):
        valid = train.iloc[valid_index]
        vector, knowbot = baseline_scores(valid)
        for name, scores in (("vector_baseline", vector), ("knowbot_baseline", knowbot)):
            records.append({"model": name, "params": "{}", "fold": fold, "split": "validation",
                            **evaluate(valid, scores.to_numpy())})

    for name, grid in SEARCH_SPACE.items():
        for params in grid:
            started = time.perf_counter()
            for fold, (fit_index, valid_index) in enumerate(folds, start=1):
                fit, valid = train.iloc[fit_index], train.iloc[valid_index]
                model = make_model(name, params, SEED).fit(fit[FEATURES], fit.label)
                for split, part in (("train", fit), ("validation", valid)):
                    scores = model.predict_proba(part[FEATURES])[:, 1]
                    records.append({"model": name, "params": json.dumps(params), "fold": fold, "split": split,
                                    **evaluate(part, scores)})
            print(f"{name:<20} {json.dumps(params):<85} {time.perf_counter() - started:5.1f}s")
    return pd.DataFrame(records)


def summarize(results):
    metrics = ["mrr", "hit_at_1", "hit_at_3", "hit_at_5", "near_hit_at_1", "near_hit_at_3", "roc_auc", "pr_auc"]
    summary = results.groupby(["model", "params", "split"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def plot_comparison(best_rows, path):
    metrics = [("mrr", "MRR"), ("hit_at_1", "Hit@1"), ("hit_at_3", "Hit@3"), ("near_hit_at_1", "Page ±1 @1")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.16
    for offset, row in enumerate(best_rows.itertuples()):
        values = [getattr(row, f"{m}_mean") for m, _ in metrics]
        errors = [getattr(row, f"{m}_std") for m, _ in metrics]
        positions = np.arange(len(metrics)) + (offset - (len(best_rows) - 1) / 2) * width
        ax.bar(positions, values, width, yerr=errors, capsize=2, label=row.model.replace("_", " "),
               color=COLORS[row.model])
    ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean over 5 validation folds")
    ax.set_title("Cross-validation: best setting of each model vs. baselines")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fit_gap(summary, best_params, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(SEARCH_SPACE)
    for split, color in (("train", "#b9c2bb"), ("validation", "#007a3d")):
        values = [summary[(summary.model == n) & (summary.params == best_params[n]) & (summary.split == split)]
                  .mrr_mean.iloc[0] for n in names]
        offset = -0.18 if split == "train" else 0.18
        ax.bar(np.arange(len(names)) + offset, values, 0.36, label=split, color=color)
    ax.set_xticks(np.arange(len(names)), [n.replace("_", " ") for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("MRR")
    ax.set_title("Training vs. validation MRR (overfitting check)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(importance, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importance = importance.sort_values("importance_mean")
    ax.barh(importance.feature, importance.importance_mean, xerr=importance.importance_std, color="#007a3d")
    ax.set_xlabel("Drop in average precision when the feature is shuffled (test set)")
    ax.set_title("Permutation importance of the selected model")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rank_of_expected(group, scores):
    ordered = group.assign(s=np.asarray(scores)).sort_values("s", ascending=False)
    positions = np.flatnonzero(ordered.label.to_numpy())
    return (int(positions[0] + 1) if len(positions) else None), ordered


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    data = pd.read_csv(os.path.join(DATA, "dataset.csv"))
    with open(os.path.join(DATA, "split.json"), encoding="utf-8") as split_file:
        split = json.load(split_file)
    train = data[data.question_id.isin(split["train_cv_question_ids"])].reset_index(drop=True)
    test = data[data.question_id.isin(split["test_question_ids"])].reset_index(drop=True)

    results = cross_validate(train)
    results.to_csv(os.path.join(REPORTS, "cv_folds.csv"), index=False)
    summary = summarize(results)
    summary.to_csv(os.path.join(REPORTS, "cv_summary.csv"), index=False)

    validation = summary[summary.split == "validation"]
    best_params = {
        name: validation[validation.model == name].sort_values("mrr_mean", ascending=False).params.iloc[0]
        for name in SEARCH_SPACE
    }
    best_rows = pd.concat(
        [validation[validation.model.isin(["vector_baseline", "knowbot_baseline"])]]
        + [validation[(validation.model == n) & (validation.params == p)] for n, p in best_params.items()]
    )
    best_rows = best_rows.set_index("model").loc[list(COLORS)].reset_index()
    best_rows.to_csv(os.path.join(REPORTS, "cv_best.csv"), index=False)
    plot_comparison(best_rows, os.path.join(REPORTS, "cv_comparison.png"))
    plot_fit_gap(summary, best_params, os.path.join(REPORTS, "fit_gap.png"))

    learned = best_rows[best_rows.model.isin(list(SEARCH_SPACE))]
    selected = learned.sort_values("mrr_mean", ascending=False).iloc[0]
    selected_name, selected_params = selected.model, json.loads(selected.params)
    print(f"Selected: {selected_name} {selected_params} (validation MRR {selected.mrr_mean:.3f})")

    # The test questions are used only here, once, after the model was selected.
    final = make_model(selected_name, selected_params, SEED).fit(train[FEATURES], train.label)
    test_scores = final.predict_proba(test[FEATURES])[:, 1]
    vector, knowbot = baseline_scores(test)
    test_metrics = {
        "selected_model": evaluate(test, test_scores),
        "vector_baseline": evaluate(test, vector.to_numpy()),
        "knowbot_baseline": evaluate(test, knowbot.to_numpy()),
        "selected_model_on_training_data": evaluate(train, final.predict_proba(train[FEATURES])[:, 1]),
    }

    questions = pd.read_csv(os.path.join(DATA, "validation_questions.csv"), encoding="utf-8-sig").set_index("id")
    questions["expected_document"] = questions["expected_document"].map(DOCUMENT_NAMES)
    per_question, per_document = [], {}
    test_scored = test.assign(score=test_scores, vector=vector.to_numpy(), knowbot=knowbot.to_numpy())
    for qid, group in test_scored.groupby("question_id"):
        rank_model, ordered = rank_of_expected(group, group.score)
        rank_vector, _ = rank_of_expected(group, group.vector)
        rank_knowbot, _ = rank_of_expected(group, group.knowbot)
        document = questions.loc[qid, "expected_document"]
        per_question.append({
            "question_id": int(qid),
            "question": questions.loc[qid, "question"],
            "expected": f"{document} p.{int(questions.loc[qid, 'expected_page'])}",
            "rank_selected_model": rank_model,
            "rank_knowbot_baseline": rank_knowbot,
            "rank_vector_search": rank_vector,
            "top1_selected_model": f"{ordered.document.iloc[0]} p.{int(ordered.page.iloc[0])}",
        })
        per_document.setdefault(document, []).append(qid)
    pd.DataFrame(per_question).to_csv(os.path.join(REPORTS, "test_per_question.csv"), index=False)
    test_metrics["selected_model_by_document"] = [
        {"document": document, "questions": len(ids),
         **ranking_metrics(test_scored[test_scored.question_id.isin(ids)], "score")}
        for document, ids in per_document.items()
    ]

    importance = permutation_importance(final, test[FEATURES], test.label, scoring="average_precision",
                                        n_repeats=20, random_state=SEED)
    importance = pd.DataFrame({"feature": FEATURES, "importance_mean": importance.importances_mean,
                               "importance_std": importance.importances_std})
    importance.to_csv(os.path.join(REPORTS, "feature_importance.csv"), index=False)
    plot_importance(importance, os.path.join(REPORTS, "feature_importance.png"))

    with open(os.path.join(REPORTS, "test_metrics.json"), "w", encoding="utf-8") as report:
        json.dump(test_metrics, report, indent=2)

    joblib.dump(final, os.path.join(MODELS, "reranker.joblib"))
    card = {
        "model": selected_name,
        "params": selected_params,
        "features": FEATURES,
        "trained_on_questions": len(split["train_cv_question_ids"]),
        "training_rows": int(len(train)),
        "selection_criterion": "highest mean validation MRR over 5 grouped folds",
        "cv_validation_mrr_mean": float(selected.mrr_mean),
        "cv_validation_mrr_std": float(selected.mrr_std),
        "test_metrics": test_metrics["selected_model"],
        "seed": SEED,
        "scikit_learn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(MODELS, "model_card.json"), "w", encoding="utf-8") as card_file:
        json.dump(card, card_file, indent=2)
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
