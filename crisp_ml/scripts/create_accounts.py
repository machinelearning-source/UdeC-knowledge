"""Create or reset the KnowBot Desk test accounts (one per role, plus a second student).

Passwords come from environment variables when they are set (DESK_COORDINATOR_PASSWORD,
DESK_ADVISOR_PASSWORD, DESK_STUDENT_PASSWORD); otherwise random passwords are generated.
They are written only to instance/test_accounts.txt, which is not committed, so they can be
given privately to the instructor.

Usage:  python scripts/create_accounts.py            (uses DATABASE_URL, or the local SQLite file)
"""

import os
import secrets
import string
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("KNOWBOT_SKIP_WARMUP", "1")

from flask import Flask  # noqa: E402

from desk import init_desk  # noqa: E402
from desk.models import ADVISOR, COORDINATOR, STUDENT, User, db  # noqa: E402

ACCOUNTS = [
    ("coordinator", "Coordinator (test)", "coordinator@knowbot-desk.test", COORDINATOR),
    ("advisor", "Advisor (test)", "advisor@knowbot-desk.test", ADVISOR),
    ("student", "Student (test)", "student@knowbot-desk.test", STUDENT),
    ("student2", "Second student (test)", "student2@knowbot-desk.test", STUDENT),
]


def random_password():
    alphabet = string.ascii_letters + string.digits
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(14))
        if any(c.isdigit() for c in password) and any(c.isalpha() for c in password):
            return password


def main():
    app = Flask(__name__)
    init_desk(app)
    lines = ["KnowBot Desk test accounts (private: give them only to the instructor)", ""]
    with app.app_context():
        for key, name, email, role in ACCOUNTS:
            password = os.environ.get(f"DESK_{key.upper()}_PASSWORD") or random_password()
            user = User.query.filter_by(email=email).first() or User(email=email)
            user.name, user.role, user.active = name, role, True
            user.set_password(password)
            db.session.add(user)
            lines.append(f"{role:<12} {email:<34} {password}")
        db.session.commit()
    path = os.path.join(ROOT, "instance", "test_accounts.txt")
    with open(path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    print(f"{len(ACCOUNTS)} accounts ready; credentials written to {path}")


if __name__ == "__main__":
    main()
