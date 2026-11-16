"""Routes of KnowBot Desk: authentication, inquiries, users, categories, reports and inference history."""

import csv
import io
import json
import statistics
from functools import wraps
from urllib.parse import urlparse

from flask import Blueprint, Response, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from . import rules
from .models import (ADVISOR, ANSWERED, CLOSED, COORDINATOR, ESCALATED, IN_PROGRESS, NEW, ROLES, STATUSES, STUDENT,
                     AuditEntry, Category, Inference, Inquiry, User, db, now)
from .rules import RuleError, audit

desk = Blueprint("desk", __name__, url_prefix="/desk")


def roles_required(*roles):
    """Server-side authorization: the route answers 403 to any other role."""
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


def _safe_next(target):
    return target if target and not urlparse(target).netloc and target.startswith("/") else None


def _inquiry_or_404(inquiry_id):
    inquiry = db.session.get(Inquiry, inquiry_id) or abort(404)
    if not rules.can_view(current_user, inquiry):
        abort(403)
    return inquiry


# ----------------------------------------------------------------- authentication

@desk.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("desk.home"))
    form = {"name": "", "email": ""}
    if request.method == "POST":
        form = {"name": request.form.get("name", ""), "email": request.form.get("email", "")}
        try:
            name, email = rules.check_registration(form["name"], form["email"], request.form.get("password", ""),
                                                   request.form.get("confirm", ""))
            if User.query.filter_by(email=email).first():
                raise RuleError("That email is already registered. Sign in instead.")
            # Every new account is a student; only a coordinator can grant another role.
            user = User(name=name, email=email, role=STUDENT)
            user.set_password(request.form["password"])
            db.session.add(user)
            db.session.flush()
            audit(user, "register", "user", user.id, "self-registration as student")
            db.session.commit()
            login_user(user)
            flash("Welcome to KnowBot Desk. Your account was created with the Student role.", "good")
            return redirect(url_for("desk.home"))
        except RuleError as error:
            flash(str(error), "bad")
    return render_template("desk/register.html", form=form)


@desk.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("desk.home"))
    email = ""
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(request.form.get("password", "")) and user.active:
            login_user(user)
            audit(user, "login", "user", user.id)
            db.session.commit()
            return redirect(_safe_next(request.args.get("next")) or url_for("desk.home"))
        flash("Wrong email or password, or the account is disabled.", "bad")
    return render_template("desk/login.html", email=email)


@desk.route("/logout", methods=["POST"])
@login_required
def logout():
    audit(current_user, "logout", "user", current_user.id)
    db.session.commit()
    logout_user()
    flash("You signed out.", "good")
    return redirect(url_for("overview"))


# ----------------------------------------------------------------- dashboard and inquiries

@desk.route("/")
@login_required
def home():
    query = Inquiry.query.order_by(Inquiry.updated_at.desc())
    if current_user.role == STUDENT:
        mine = query.filter_by(student_id=current_user.id).all()
        return render_template("desk/home_student.html", inquiries=mine)
    queue = query.filter_by(status=NEW).all()
    mine = query.filter_by(advisor_id=current_user.id, status=IN_PROGRESS).all()
    escalated = query.filter_by(status=ESCALATED).all()
    return render_template("desk/home_staff.html", queue=queue, mine=mine, escalated=escalated,
                           indicators=_indicators())


@desk.route("/inquiries")
@login_required
def inquiries():
    query = Inquiry.query.order_by(Inquiry.created_at.desc())
    if current_user.role == STUDENT:
        query = query.filter_by(student_id=current_user.id)
    status, category = request.args.get("status"), request.args.get("category", type=int)
    if status in STATUSES:
        query = query.filter_by(status=status)
    if category:
        query = query.filter_by(category_id=category)
    return render_template("desk/inquiries.html", inquiries=query.all(), statuses=STATUSES,
                           categories=Category.query.order_by(Category.name).all(), status=status, category=category)


@desk.route("/inquiries/new", methods=["GET", "POST"])
@roles_required(STUDENT)
def new_inquiry():
    categories = Category.query.filter_by(active=True).order_by(Category.name).all()
    form = {"title": "", "question": "", "category_id": None}
    if request.method == "POST":
        form = {"title": request.form.get("title", ""), "question": request.form.get("question", ""),
                "category_id": request.form.get("category_id", type=int)}
        try:
            title = rules.check_length("The title", form["title"], rules.TITLE_CHARS)
            question = rules.check_length("The question", form["question"], rules.QUESTION_CHARS)
            category = db.session.get(Category, form["category_id"] or 0)
            if not category or not category.active:
                raise RuleError("Choose a category.")
            inquiry = Inquiry(student_id=current_user.id, category=category, title=title, question=question)
            db.session.add(inquiry)
            db.session.flush()
            rules.run_model(inquiry, current_user)
            audit(current_user, "create", "inquiry", inquiry.id, f"status {inquiry.status}")
            db.session.commit()
            flash("Your inquiry was registered." + (
                " The model was not sure about the source, so a coordinator will review it."
                if inquiry.status == ESCALATED else " An advisor will answer it."), "good")
            return redirect(url_for("desk.inquiry", inquiry_id=inquiry.id))
        except RuleError as error:
            flash(str(error), "bad")
    return render_template("desk/inquiry_form.html", form=form, categories=categories, editing=False)


@desk.route("/inquiries/<int:inquiry_id>")
@login_required
def inquiry(inquiry_id):
    item = _inquiry_or_404(inquiry_id)
    latest = item.inferences[-1] if item.inferences else None
    ranking = json.loads(latest.ranking) if latest and latest.ranking else []
    return render_template("desk/inquiry.html", item=item, latest=latest, ranking=ranking, rules=rules,
                           user=current_user)


@desk.route("/inquiries/<int:inquiry_id>/edit", methods=["GET", "POST"])
@login_required
def edit_inquiry(inquiry_id):
    item = _inquiry_or_404(inquiry_id)
    if not rules.can_edit(current_user, item):
        abort(403)
    categories = Category.query.filter_by(active=True).order_by(Category.name).all()
    form = {"title": item.title, "question": item.question, "category_id": item.category_id}
    if request.method == "POST":
        form = {"title": request.form.get("title", ""), "question": request.form.get("question", ""),
                "category_id": request.form.get("category_id", type=int)}
        try:
            item.title = rules.check_length("The title", form["title"], rules.TITLE_CHARS)
            question = rules.check_length("The question", form["question"], rules.QUESTION_CHARS)
            category = db.session.get(Category, form["category_id"] or 0)
            if not category or not category.active:
                raise RuleError("Choose a category.")
            item.category = category
            if question != item.question:
                item.question = question
                rules.run_model(item, current_user)  # a new question gets a new prediction
            audit(current_user, "update", "inquiry", item.id)
            db.session.commit()
            flash("The inquiry was updated.", "good")
            return redirect(url_for("desk.inquiry", inquiry_id=item.id))
        except RuleError as error:
            flash(str(error), "bad")
    return render_template("desk/inquiry_form.html", form=form, categories=categories, editing=True, item=item)


@desk.route("/inquiries/<int:inquiry_id>/delete", methods=["POST"])
@login_required
def delete_inquiry(inquiry_id):
    item = _inquiry_or_404(inquiry_id)
    if not (rules.can_edit(current_user, item) or current_user.role == COORDINATOR):
        abort(403)
    audit(current_user, "delete", "inquiry", item.id, item.title)
    db.session.delete(item)
    db.session.commit()
    flash("The inquiry was deleted.", "good")
    return redirect(url_for("desk.home"))


@desk.route("/inquiries/<int:inquiry_id>/take", methods=["POST"])
@roles_required(ADVISOR, COORDINATOR)
def take_inquiry(inquiry_id):
    item = _inquiry_or_404(inquiry_id)
    if not rules.can_take(current_user, item):
        flash("This inquiry cannot be taken in its current state.", "bad")
        return redirect(url_for("desk.inquiry", inquiry_id=item.id))
    item.advisor_id, item.status = current_user.id, IN_PROGRESS
    audit(current_user, "take", "inquiry", item.id)
    db.session.commit()
    return redirect(url_for("desk.inquiry", inquiry_id=item.id))


@desk.route("/inquiries/<int:inquiry_id>/escalate", methods=["POST"])
@roles_required(ADVISOR)
def escalate_inquiry(inquiry_id):
    item = _inquiry_or_404(inquiry_id)
    if item.status not in (NEW, IN_PROGRESS):
        abort(403)
    item.status, item.advisor_id = ESCALATED, None
    audit(current_user, "escalate", "inquiry", item.id, "sent to the coordinator by an advisor")
    db.session.commit()
    flash("The inquiry was sent to the coordinator.", "good")
    return redirect(url_for("desk.home"))


@desk.route("/inquiries/<int:inquiry_id>/answer", methods=["POST"])
@roles_required(ADVISOR, COORDINATOR)
def answer_inquiry(inquiry_id):
    item = _inquiry_or_404(inquiry_id)
    if not rules.can_answer(current_user, item):
        abort(403)
    try:
        answer = rules.check_length("The answer", request.form.get("answer"), rules.ANSWER_CHARS)
        source = request.form.get("source", "")
        if source == "other":
            document = rules.check_length("The document", request.form.get("other_document"), (3, 120))
            page = request.form.get("other_page", type=int)
            if not page or page < 1:
                raise RuleError("Write the page number of the source.")
        elif "|" in source:
            document, page = source.rsplit("|", 1)
            page = int(page)
        else:
            raise RuleError("Choose the page that supports the answer.")
        item.answer, item.source_document, item.source_page = answer, document, page
        item.status, item.answered_at = ANSWERED, now()
        audit(current_user, "answer", "inquiry", item.id,
              f"source {document} p.{page}; model {'right' if item.model_was_right else 'corrected'}")
        db.session.commit()
        flash("The answer was sent to the student.", "good")
    except (RuleError, ValueError) as error:
        flash(str(error), "bad")
    return redirect(url_for("desk.inquiry", inquiry_id=item.id))


@desk.route("/inquiries/<int:inquiry_id>/rate", methods=["POST"])
@roles_required(STUDENT)
def rate_inquiry(inquiry_id):
    item = _inquiry_or_404(inquiry_id)
    if not rules.can_rate(current_user, item):
        abort(403)
    rating = request.form.get("rating", type=int)
    if rating not in (1, 2, 3, 4, 5):
        flash("Choose a rating from 1 to 5.", "bad")
        return redirect(url_for("desk.inquiry", inquiry_id=item.id))
    item.rating, item.rating_comment = rating, (request.form.get("comment") or "").strip()[:500]
    item.status, item.closed_at = CLOSED, now()
    audit(current_user, "close", "inquiry", item.id, f"rating {rating}")
    db.session.commit()
    flash("Thank you. The inquiry is closed.", "good")
    return redirect(url_for("desk.inquiry", inquiry_id=item.id))


# ----------------------------------------------------------------- administration (coordinator)

@desk.route("/users")
@roles_required(COORDINATOR)
def users():
    return render_template("desk/users.html", users=User.query.order_by(User.created_at).all(), roles=ROLES)


@desk.route("/users/<int:user_id>/role", methods=["POST"])
@roles_required(COORDINATOR)
def change_role(user_id):
    target = db.session.get(User, user_id) or abort(404)
    try:
        new_role = request.form.get("role")
        rules.check_role_change(current_user, target, new_role)
        audit(current_user, "change role", "user", target.id, f"{target.role} -> {new_role}")
        target.role = new_role
        db.session.commit()
        flash(f"{target.name} is now {ROLES[new_role]}.", "good")
    except RuleError as error:
        flash(str(error), "bad")
    return redirect(url_for("desk.users"))


@desk.route("/users/<int:user_id>/active", methods=["POST"])
@roles_required(COORDINATOR)
def toggle_user(user_id):
    target = db.session.get(User, user_id) or abort(404)
    if target.id == current_user.id:
        flash("You cannot disable your own account.", "bad")
    else:
        target.active = not target.active
        audit(current_user, "enable" if target.active else "disable", "user", target.id)
        db.session.commit()
    return redirect(url_for("desk.users"))


@desk.route("/categories", methods=["GET", "POST"])
@roles_required(COORDINATOR)
def categories():
    if request.method == "POST":
        try:
            name = rules.check_length("The name", request.form.get("name"), (3, 80))
            description = (request.form.get("description") or "").strip()[:300]
            if Category.query.filter(db.func.lower(Category.name) == name.lower()).first():
                raise RuleError("A category with that name already exists.")
            db.session.add(Category(name=name, description=description))
            audit(current_user, "create", "category", None, name)
            db.session.commit()
            flash("Category created.", "good")
        except RuleError as error:
            flash(str(error), "bad")
        return redirect(url_for("desk.categories"))
    rows = [(c, Inquiry.query.filter_by(category_id=c.id).count()) for c in Category.query.order_by(Category.name)]
    return render_template("desk/categories.html", rows=rows)


@desk.route("/categories/<int:category_id>", methods=["POST"])
@roles_required(COORDINATOR)
def update_category(category_id):
    category = db.session.get(Category, category_id) or abort(404)
    action = request.form.get("action")
    if action == "delete":
        if Inquiry.query.filter_by(category_id=category.id).count():
            flash("A category with inquiries cannot be deleted; disable it instead.", "bad")
        else:
            audit(current_user, "delete", "category", category.id, category.name)
            db.session.delete(category)
            db.session.commit()
            flash("Category deleted.", "good")
    elif action == "toggle":
        category.active = not category.active
        audit(current_user, "enable" if category.active else "disable", "category", category.id)
        db.session.commit()
    else:
        try:
            category.name = rules.check_length("The name", request.form.get("name"), (3, 80))
            category.description = (request.form.get("description") or "").strip()[:300]
            audit(current_user, "update", "category", category.id)
            db.session.commit()
            flash("Category updated.", "good")
        except RuleError as error:
            db.session.rollback()
            flash(str(error), "bad")
    return redirect(url_for("desk.categories"))


# ----------------------------------------------------------------- reports and model monitoring

def _indicators():
    all_items = Inquiry.query.all()
    answered = [i for i in all_items if i.answered_at]
    checked = [i for i in answered if i.model_was_right is not None]
    hours = [(i.answered_at - i.created_at).total_seconds() / 3600 for i in answered]
    ratings = [i.rating for i in all_items if i.rating]
    inferences = Inference.query.all()
    latencies = [x.latency_ms for x in inferences if x.latency_ms]
    return {
        "total": len(all_items),
        "by_status": {name: sum(i.status == key for i in all_items) for key, name in STATUSES.items()},
        "by_category": sorted(((c.name, len(c.inquiries)) for c in Category.query.all()), key=lambda x: -x[1]),
        "median_hours": round(statistics.median(hours), 1) if hours else None,
        "mean_rating": round(statistics.mean(ratings), 2) if ratings else None,
        "model_checked": len(checked),
        "model_right": sum(i.model_was_right for i in checked),
        "model_accuracy": round(sum(i.model_was_right for i in checked) / len(checked), 3) if checked else None,
        "escalated_by_model": sum(x.decision == "escalate to coordinator" for x in inferences),
        "inferences": len(inferences),
        "model_errors": sum(bool(x.error) for x in inferences),
        "median_latency": round(statistics.median(latencies), 1) if latencies else None,
    }


@desk.route("/reports")
@roles_required(ADVISOR, COORDINATOR)
def reports():
    return render_template("desk/reports.html", indicators=_indicators(),
                           audit=AuditEntry.query.order_by(AuditEntry.created_at.desc()).limit(40).all())


@desk.route("/inferences")
@roles_required(ADVISOR, COORDINATOR)
def inferences():
    rows = Inference.query.order_by(Inference.created_at.desc()).limit(200).all()
    return render_template("desk/inferences.html", rows=rows)


@desk.route("/reports/labeled.csv")
@roles_required(COORDINATOR)
def labeled_csv():
    """Questions with the page confirmed by an advisor: new labels to retrain and re-test the model."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["inquiry_id", "question", "category", "suggested_document", "suggested_page",
                     "suggested_probability", "confirmed_document", "confirmed_page", "model_was_right", "rating"])
    for i in Inquiry.query.filter(Inquiry.source_page.isnot(None)).order_by(Inquiry.id):
        writer.writerow([i.id, i.question, i.category.name, i.suggested_document, i.suggested_page,
                         i.suggested_probability, i.source_document, i.source_page, i.model_was_right, i.rating])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=knowbot_desk_labeled_questions.csv"})
