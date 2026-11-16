"""KnowBot Desk: the private web information system that manages academic inquiries.

Registration, login and roles protect every operational route; the public pages of the
application (PGC documentation, the six CRISP-ML phases and the demo) stay open.
"""

import os
import secrets

from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from .models import Category, User, db

DEFAULT_CATEGORIES = [
    ("Admissions and registration", "Registration, enrollment, re-entry, transfers and academic registration."),
    ("Course homologation", "Recognition of courses taken in other programs or universities."),
    ("Academic calendar", "Dates of the 2026 academic calendar activities."),
    ("Student situations", "Rights, duties and procedures for specific student situations."),
    ("Formative experiences", "Agreement 005 of 2025: proposals, committee and requirements."),
    ("Discipline and sanctions", "Disciplinary regime, faults, sanctions and procedures."),
]

login_manager = LoginManager()
csrf = CSRFProtect()


def init_desk(app):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    database_url = os.environ.get("DATABASE_URL", "sqlite:///" + os.path.join(root, "instance", "knowbot_desk.db"))
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]
    os.makedirs(os.path.join(root, "instance"), exist_ok=True)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=database_url,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")),
        WTF_CSRF_TIME_LIMIT=None,
    )
    db.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "desk.login"
    login_manager.login_message = "Sign in to use KnowBot Desk."

    from .views import desk
    app.register_blueprint(desk)

    with app.app_context():
        db.create_all()
        if not Category.query.first():
            db.session.add_all(Category(name=name, description=text) for name, text in DEFAULT_CATEGORIES)
            db.session.commit()
        _create_test_accounts()


# Test accounts for the instructor. Each one is created at start-up only when its password is set
# in the environment of the server (never in the code), and only if it does not exist yet.
TEST_ACCOUNTS = [
    ("COORDINATOR", "Coordinator (test)", "coordinator@knowbot-desk.test", "coordinator"),
    ("ADVISOR", "Advisor (test)", "advisor@knowbot-desk.test", "advisor"),
    ("STUDENT", "Student (test)", "student@knowbot-desk.test", "student"),
]


def _create_test_accounts():
    created = False
    for key, name, email, role in TEST_ACCOUNTS:
        password = os.environ.get(f"DESK_{key}_PASSWORD")
        if password and not User.query.filter_by(email=email).first():
            user = User(name=name, email=email, role=role)
            user.set_password(password)
            db.session.add(user)
            created = True
    if created:
        db.session.commit()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))
