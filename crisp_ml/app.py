"""KnowBot Re-ranker: CRISP-ML documentation and model demonstration (Flask).

Every figure shown in the phase pages is read from the files written by
scripts/build_dataset.py and scripts/train.py, so the application, the
repository and the technical report always show the same results.

Run locally:  python app.py   (then open http://127.0.0.1:5000)
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

# One thread per numerical library. The free hosting instance has a 0.1 CPU quota but shows
# the host's many cores; OpenMP and BLAS thread pools sized for those cores waste the quota.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
# The model runs in this process only; joblib's process pool and its semaphores are not needed.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
# The embedding model and its tokenizer are downloaded during the build. At run time they are
# read from that cache only: no request to the model hub, which could retry and wait for minutes.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# All heavy libraries are imported here, in the main thread, before the warm-up thread starts.
# On the Linux host, importing joblib from the background thread blocked, so the model never
# finished loading and every later use of joblib found it partially initialized.
import faiss  # noqa: E402, F401
import joblib  # noqa: E402, F401
import onnxruntime  # noqa: E402, F401
import pandas as pd  # noqa: E402
import sentencepiece  # noqa: E402, F401
import sklearn.ensemble  # noqa: E402, F401
from fastembed import TextEmbedding  # noqa: E402, F401
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402, F401
from langchain_community.vectorstores import FAISS  # noqa: E402, F401

from knowbot import monitoring, predict, quick_questions  # noqa: E402, F401
from knowbot.answer import write_answer  # noqa: E402
from desk import csrf, init_desk  # noqa: E402
from desk.models import COORDINATOR  # noqa: E402
from desk.views import roles_required  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
DATA = os.path.join(ROOT, "data")

app = Flask(__name__)


@app.template_filter("simple_markdown")
def simple_markdown(text):
    """Bold, line breaks and list markers of a written answer, with everything else escaped."""
    import html
    import re

    from markupsafe import Markup

    lines = []
    for line in html.escape(text or "").splitlines():
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"【(\d+)[^】]*】", r"[\1]", line)
        line = re.sub(r"^\s*[-*•]\s+", "• ", line)
        lines.append(line)
    return Markup("<br>".join(lines))

PHASES = [
    ("business", "1. Business and Data Understanding"),
    ("data", "2. Data Engineering"),
    ("modeling", "3. Model Engineering"),
    ("evaluation", "4. Model Evaluation"),
    ("deployment", "5. Model Deployment"),
    ("monitoring", "6. Monitoring and Maintenance"),
]

_model_ready = threading.Event()
_warm_up_lock = threading.Lock()
_warm_up_pid = None


def _warm_up():
    """Load the embedding model, the index and the re-ranker in the background after start-up."""
    import faulthandler
    import sys

    # If loading takes more than 3 minutes, write where every thread is to the log.
    faulthandler.dump_traceback_later(180, file=sys.stderr)
    started = time.perf_counter()
    try:
        from knowbot.predict import rerank

        rerank("When does online registration take place in 2026?")
        print(f"Model loaded in {time.perf_counter() - started:.1f} s", file=sys.stderr, flush=True)
    except Exception as error:
        print(f"Model loading failed: {error!r}", file=sys.stderr, flush=True)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        _model_ready.set()


def _json(name):
    with open(os.path.join(REPORTS, name), encoding="utf-8") as report:
        return json.load(report)


def _csv(path):
    return pd.read_csv(path)


@app.context_processor
def navigation():
    return {"phases": PHASES, "current_year": datetime.now().year, "resources": _resources()}


FINAL_REPORT = os.path.join(ROOT, "static", "docs", "KnowBot_Final_Technical_Report.pdf")


def _resources():
    """Links shown on the homepage: the final report once it is published, and the presentation video."""
    report_url = (url_for("static", filename="docs/KnowBot_Final_Technical_Report.pdf") if os.path.exists(FINAL_REPORT)
                  else "https://github.com/machinelearning-source/UdeC-knowledge/tree/main/crisp_ml/reports")
    return {"report_url": report_url, "video_url": os.environ.get("PROJECT_VIDEO_URL", "")}


@app.route("/guide")
def guide():
    """User documentation: how each role uses KnowBot Desk, plus incident reporting and feedback."""
    return render_template("guide.html")


@app.route("/")
def overview():
    card = json.load(open(os.path.join(ROOT, "models", "model_card.json"), encoding="utf-8"))
    test = _json("test_metrics.json")
    quality = _json("data_quality.json")
    return render_template("overview.html", card=card, test=test, quality=quality)


@app.route("/phase/<slug>")
def phase(slug):
    if slug not in dict(PHASES):
        abort(404)
    context = {"slug": slug, "title": dict(PHASES)[slug]}
    context["quality"] = _json("data_quality.json")
    context["card"] = json.load(open(os.path.join(ROOT, "models", "model_card.json"), encoding="utf-8"))
    if slug == "data":
        dataset = _csv(os.path.join(DATA, "dataset.csv"))
        raw = _csv(os.path.join(DATA, "raw_candidates.csv"))
        context["raw_sample"] = raw.drop(columns=["text"]).head(6).round(3).to_dict("records")
        context["raw_columns"] = list(raw.drop(columns=["text"]).columns)
        context["describe"] = dataset[["cosine", "bm25", "bm25_rank", "chunk_chars", "word_overlap"]].describe().round(3)
        context["processed"] = _processed_sample(dataset)
        context["split"] = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    if slug in ("modeling", "evaluation"):
        context["best"] = _csv(os.path.join(REPORTS, "cv_best.csv")).to_dict("records")
        context["experiments"] = _experiments()
        summary = _csv(os.path.join(REPORTS, "cv_summary.csv"))
        context["grid"] = summary[summary.split == "validation"].sort_values(
            ["model", "mrr_mean"], ascending=[True, False]).to_dict("records")
        context["train_rows"] = summary[summary.split == "train"].set_index(["model", "params"]).mrr_mean.to_dict()
    if slug == "evaluation":
        context["test"] = _json("test_metrics.json")
        context["per_question"] = _csv(os.path.join(REPORTS, "test_per_question.csv")).to_dict("records")
        context["importance"] = _csv(os.path.join(REPORTS, "feature_importance.csv")).sort_values(
            "importance_mean", ascending=False).to_dict("records")
        review_path = os.path.join(REPORTS, "team_review.json")
        context["review"] = json.load(open(review_path, encoding="utf-8")) if os.path.exists(review_path) else None
    if slug == "monitoring":
        context["log"] = monitoring.read_log(limit=50)
        context["indicators"] = monitoring.indicators()
    return render_template(f"phase_{slug}.html", **context)


def _processed_sample(dataset):
    """The first rows after the training-fitted preprocessing (imputation, log scaling, standardization, one-hot)."""
    import joblib

    from knowbot.features import FEATURES

    model = joblib.load(os.path.join(ROOT, "models", "reranker.joblib"))
    preprocess = model.named_steps["preprocess"]
    values = preprocess.transform(dataset[FEATURES].head(6))
    names = [name.split("__", 1)[1] for name in preprocess.get_feature_names_out()]
    return {"columns": names, "rows": [[round(float(v), 3) for v in row] for row in values]}


def _experiments():
    experiments = []
    base = os.path.join(REPORTS, "experiments")
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        best = pd.read_csv(os.path.join(base, name, "cv_best.csv"))
        notes_path = os.path.join(base, name, "notes.json")
        notes = json.load(open(notes_path, encoding="utf-8")) if os.path.exists(notes_path) else {}
        experiments.append({"name": name, "notes": notes, "rows": best.to_dict("records")})
    return experiments


@app.route("/demo", methods=["GET", "POST"])
def demo():
    result, error, question = None, None, ""
    if request.method == "POST":
        question = _quick_question(request.form) or request.form.get("question", "")
        result, error = _predict(question, source="demo")
        if result:
            result["answer"], result["answer_error"] = write_answer(result["question"], result["answer_fragments"])
    return render_template("demo.html", result=result, error=error, question=question,
                           ready=_model_ready.is_set(), examples=_examples(), quick=QUICK_MENUS)


# The three menus of the KnowBot chatbot: choose an option, then confirm it.
QUICK_MENUS = [
    {"name": "category", "title": "Student categories", "choose": "Choose a student situation…",
     "button": "Ask about this category", "options": quick_questions.STUDENT_CATEGORIES},
    {"name": "course", "title": "Course homologation", "choose": "Choose a course…",
     "button": "Ask about this course", "options": quick_questions.HOMOLOGATION_COURSES},
    {"name": "calendar", "title": "Academic calendar", "choose": "Choose an activity…",
     "button": "Ask about this date", "options": quick_questions.CALENDAR_ACTIVITIES},
]
QUICK_TEMPLATES = {"category": quick_questions.CATEGORY_QUESTION, "course": quick_questions.COURSE_QUESTION,
                   "calendar": quick_questions.CALENDAR_QUESTION}


def _quick_question(form):
    """The question of a confirmed quick menu, or None when the question box was used."""
    menu = form.get("quick")
    option = form.get("option", "")
    menus = {item["name"]: item for item in QUICK_MENUS}
    if menu in menus and option in menus[menu]["options"]:
        return QUICK_TEMPLATES[menu].format(option=option)
    return None


def _examples():
    split = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    questions = pd.read_csv(os.path.join(DATA, "validation_questions.csv"), encoding="utf-8-sig").set_index("id")
    return [questions.loc[qid, "question"] for qid in split["test_question_ids"][:4]]


def _predict(question, source):
    from knowbot.predict import rerank

    started = time.perf_counter()
    try:
        result = rerank(question)
    except ValueError as exc:
        monitoring.log_prediction(question, None, source, error=str(exc))
        return None, str(exc)
    monitoring.log_prediction(question, result, source)
    result["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result, None


@app.route("/api/rerank", methods=["POST"])
@csrf.exempt
def api_rerank():
    """JSON API: {"question": "..."} -> the five best fragments by the model and by KnowBot's order."""
    payload = request.get_json(silent=True) or {}
    result, error = _predict(payload.get("question", ""), source="api")
    if error:
        return jsonify({"error": error}), 400
    result.pop("answer_fragments", None)
    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback():
    """User feedback on a demo result: was the first fragment the right one?"""
    monitoring.log_feedback(request.form.get("question", ""), request.form.get("useful", ""),
                            request.form.get("comment", ""))
    return render_template("thanks.html")


@app.route("/monitoring/simulate", methods=["POST"])
@roles_required(COORDINATOR)
def simulate():
    """Run the 20 test questions through the live model and log them as SIMULATED traffic."""
    split = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    questions = pd.read_csv(os.path.join(DATA, "validation_questions.csv"), encoding="utf-8-sig").set_index("id")
    for qid in split["test_question_ids"]:
        _predict(questions.loc[qid, "question"], source="simulated")
    return redirect(url_for("phase", slug="monitoring"))


@app.route("/reports/<path:name>")
def report_file(name):
    return send_from_directory(REPORTS, name)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model_ready.is_set(),
                    "time": datetime.now(timezone.utc).isoformat(timespec="seconds")})


@app.before_request
def _start_warm_up():
    """Start loading the model once in each serving process, at its first request.

    gunicorn imports this module in its master process and then forks the worker. A thread
    started at import would run in the master only (threads do not survive fork), so the
    worker would never see the model loaded. Starting it here keeps it in the worker.
    """
    global _warm_up_pid
    if _warm_up_pid == os.getpid():
        return
    with _warm_up_lock:
        if _warm_up_pid != os.getpid():
            _warm_up_pid = os.getpid()
            threading.Thread(target=_warm_up, daemon=True).start()

# KnowBot Desk: registration, login, roles and the inquiry management system (desk/).
init_desk(app)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
