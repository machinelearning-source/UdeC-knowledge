"""Business rules of KnowBot Desk and the decision supported by the Machine Learning model."""

import json
import re
import time

from knowbot.predict import model_card, rerank

from .models import (ADVISOR, ANSWERED, CLOSED, COORDINATOR, ESCALATED, IN_PROGRESS, NEW, STUDENT, AuditEntry,
                     Inference, db)

# Thresholds on the probability of the page the model puts first (see Phase 4 and Phase 6).
ESCALATE_BELOW = 0.30   # the model is unsure: the inquiry goes to the coordinator
SUGGEST_FROM = 0.50     # the model is confident: its page is pre-selected for the advisor

TITLE_CHARS = (5, 120)
QUESTION_CHARS = (15, 500)
ANSWER_CHARS = (20, 4000)
PASSWORD_MIN = 8
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RuleError(ValueError):
    """A business rule was not met; the message is shown to the user."""


def model_version():
    card = model_card()
    return f"{card['model']} trained {card['trained_at'][:10]}"


def check_length(label, text, limits):
    text = (text or "").strip()
    low, high = limits
    if not low <= len(text) <= high:
        raise RuleError(f"{label} must have between {low} and {high} characters.")
    return text


def check_registration(name, email, password, confirm):
    name = check_length("Name", name, (3, 120))
    email = (email or "").strip().lower()
    if not EMAIL_PATTERN.match(email) or len(email) > 160:
        raise RuleError("Write a valid email address.")
    if len(password or "") < PASSWORD_MIN or password.isdigit() or password.isalpha():
        raise RuleError(f"The password needs at least {PASSWORD_MIN} characters, with letters and numbers.")
    if password != confirm:
        raise RuleError("The two passwords do not match.")
    return name, email


def audit(user, action, entity, entity_id=None, detail=""):
    db.session.add(AuditEntry(user_id=user.id if user else None, action=action, entity=entity,
                              entity_id=entity_id, detail=detail[:300]))


def run_model(inquiry, user):
    """Run the re-ranker for an inquiry, store the inference and apply the routing decision.

    Decision supported by the model:
    - top probability below 0.30: escalate the inquiry to the coordinator (the model is unsure);
    - otherwise: the inquiry goes to the advisors' queue with the suggested page.
    """
    started = time.perf_counter()
    try:
        result = rerank(inquiry.question)
    except Exception as error:  # the inquiry is kept; the advisor works without a suggestion
        db.session.add(Inference(inquiry=inquiry, user_id=user.id, question_chars=len(inquiry.question),
                                 decision="model error: manual review", model_name="re-ranker",
                                 model_version="unavailable", error=str(error)[:300],
                                 latency_ms=round((time.perf_counter() - started) * 1000, 1)))
        inquiry.status = NEW
        return None
    top = result["reranked"][0]
    probability = result["top_probability"]
    decision = "escalate to coordinator" if probability < ESCALATE_BELOW else "advisor queue"
    inquiry.suggested_document, inquiry.suggested_page = top["document"], top["page"]
    inquiry.suggested_probability = probability
    inquiry.status = ESCALATED if probability < ESCALATE_BELOW else NEW
    ranking = [{key: item[key] for key in ("rank", "document", "page", "probability", "excerpt")}
               for item in result["reranked"]]
    db.session.add(Inference(inquiry=inquiry, user_id=user.id, question_chars=len(inquiry.question),
                             top_document=top["document"], top_page=top["page"], top_probability=probability,
                             decision=decision, model_name=result["model"], model_version=model_version(),
                             latency_ms=result["latency_ms"], ranking=json.dumps(ranking)))
    return result


def can_view(user, inquiry):
    return user.role in (ADVISOR, COORDINATOR) or inquiry.student_id == user.id


def can_take(user, inquiry):
    """Advisors take new inquiries; escalated ones are taken by the coordinator."""
    if inquiry.status == NEW:
        return user.role in (ADVISOR, COORDINATOR)
    if inquiry.status == ESCALATED:
        return user.role == COORDINATOR
    return False


def can_answer(user, inquiry):
    return inquiry.status == IN_PROGRESS and (inquiry.advisor_id == user.id or user.role == COORDINATOR)


def can_rate(user, inquiry):
    return inquiry.status == ANSWERED and inquiry.student_id == user.id


def can_edit(user, inquiry):
    """A student can edit or delete an inquiry only while nobody has taken it."""
    return inquiry.student_id == user.id and inquiry.status in (NEW, ESCALATED)


def check_role_change(actor, target, new_role):
    if actor.role != COORDINATOR:
        raise RuleError("Only a coordinator can change roles.")
    if new_role not in (STUDENT, ADVISOR, COORDINATOR):
        raise RuleError("Unknown role.")
    if target.id == actor.id:
        raise RuleError("You cannot change your own role.")


def close_states():
    return (ANSWERED, CLOSED)
