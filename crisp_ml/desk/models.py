"""Database model of KnowBot Desk, the academic inquiry management system.

Entities: users with a role, inquiry categories, inquiries (the business process), the model
inferences made for each inquiry and an audit log of the actions that change data.
"""

from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()

STUDENT, ADVISOR, COORDINATOR = "student", "advisor", "coordinator"
ROLES = {STUDENT: "Student", ADVISOR: "Advisor", COORDINATOR: "Coordinator"}

# Inquiry life cycle: new -> (escalated) -> in_progress -> answered -> closed
NEW, ESCALATED, IN_PROGRESS, ANSWERED, CLOSED = "new", "escalated", "in_progress", "answered", "closed"
STATUSES = {NEW: "New", ESCALATED: "Escalated", IN_PROGRESS: "In progress", ANSWERED: "Answered", CLOSED: "Closed"}


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(160), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=STUDENT)
    active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=now)

    inquiries = db.relationship("Inquiry", back_populates="student", foreign_keys="Inquiry.student_id")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_active(self):
        return self.active

    @property
    def role_name(self):
        return ROLES.get(self.role, self.role)


class Category(db.Model):
    __tablename__ = "categories"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    description = db.Column(db.String(300), nullable=False, default="")
    active = db.Column(db.Boolean, nullable=False, default=True)

    inquiries = db.relationship("Inquiry", back_populates="category")


class Inquiry(db.Model):
    __tablename__ = "inquiries"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey("categories.id"), nullable=False)
    advisor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    title = db.Column(db.String(120), nullable=False)
    question = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), nullable=False, default=NEW, index=True)
    # The page the model put first, kept for the comparison with the actual outcome.
    suggested_document = db.Column(db.String(120))
    suggested_page = db.Column(db.Integer)
    suggested_probability = db.Column(db.Float)
    # The actual outcome: the page the advisor confirmed as the source of the answer.
    source_document = db.Column(db.String(120))
    source_page = db.Column(db.Integer)
    answer = db.Column(db.Text)
    rating = db.Column(db.Integer)
    rating_comment = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    updated_at = db.Column(db.DateTime, nullable=False, default=now, onupdate=now)
    answered_at = db.Column(db.DateTime)
    closed_at = db.Column(db.DateTime)

    student = db.relationship("User", foreign_keys=[student_id], back_populates="inquiries")
    advisor = db.relationship("User", foreign_keys=[advisor_id])
    category = db.relationship("Category", back_populates="inquiries")
    inferences = db.relationship("Inference", back_populates="inquiry", order_by="Inference.created_at",
                                 cascade="all, delete-orphan")

    @property
    def status_name(self):
        return STATUSES.get(self.status, self.status)

    @property
    def model_was_right(self):
        """True or False once the advisor has confirmed the source page; None before."""
        if self.source_page is None or self.suggested_page is None:
            return None
        return (self.source_document, self.source_page) == (self.suggested_document, self.suggested_page)


class Inference(db.Model):
    """Inference history: date, user, result and model version of every prediction."""
    __tablename__ = "inferences"
    id = db.Column(db.Integer, primary_key=True)
    inquiry_id = db.Column(db.Integer, db.ForeignKey("inquiries.id"), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    question_chars = db.Column(db.Integer, nullable=False)
    top_document = db.Column(db.String(120))
    top_page = db.Column(db.Integer)
    top_probability = db.Column(db.Float)
    decision = db.Column(db.String(30), nullable=False)
    model_name = db.Column(db.String(60), nullable=False)
    model_version = db.Column(db.String(60), nullable=False)
    latency_ms = db.Column(db.Float)
    ranking = db.Column(db.Text)  # JSON: the five fragments the model put first
    error = db.Column(db.String(300))

    inquiry = db.relationship("Inquiry", back_populates="inferences")
    user = db.relationship("User")


class AuditEntry(db.Model):
    __tablename__ = "audit_log"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    action = db.Column(db.String(60), nullable=False)
    entity = db.Column(db.String(40), nullable=False)
    entity_id = db.Column(db.Integer)
    detail = db.Column(db.String(300), nullable=False, default="")

    user = db.relationship("User")
