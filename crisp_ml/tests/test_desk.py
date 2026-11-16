"""End-to-end tests of KnowBot Desk: authentication, roles, the inquiry workflow and the model.

Run:  python -m unittest tests.test_desk -v     (uses a temporary SQLite database)
"""

import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DB_FILE = os.path.join(tempfile.mkdtemp(), "desk_test.db")
os.environ["DATABASE_URL"] = "sqlite:///" + DB_FILE.replace("\\", "/")

import app as application  # noqa: E402
from desk.models import ADVISOR, ANSWERED, CLOSED, COORDINATOR, Inference, Inquiry, User, db  # noqa: E402

flask_app = application.app
flask_app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
PASSWORD = "Test12345pass"


def make_user(email, role, name="Test user"):
    with flask_app.app_context():
        user = User(name=name, email=email, role=role)
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()


class DeskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_user("coord@test.edu", COORDINATOR, "Coordinator")
        make_user("advisor@test.edu", ADVISOR, "Advisor")

    def client_for(self, email, password=PASSWORD):
        client = flask_app.test_client()
        response = client.post("/desk/login", data={"email": email, "password": password})
        self.assertEqual(response.status_code, 302, f"login failed for {email}")
        return client

    def register(self, email, name="Student"):
        client = flask_app.test_client()
        response = client.post("/desk/register", data={"name": name, "email": email, "password": PASSWORD,
                                                       "confirm": PASSWORD})
        self.assertEqual(response.status_code, 302)
        return client

    def test_1_public_pages_and_protected_routes(self):
        client = flask_app.test_client()
        for path in ("/", "/phase/business", "/phase/monitoring", "/desk/login", "/desk/register", "/health"):
            self.assertEqual(client.get(path).status_code, 200, path)
        for path in ("/desk/", "/desk/inquiries", "/desk/inquiries/new", "/desk/users", "/desk/reports"):
            response = client.get(path)
            self.assertEqual(response.status_code, 302, path)
            self.assertIn("/desk/login", response.headers["Location"])
        self.assertEqual(client.post("/monitoring/simulate").status_code, 302)  # needs a coordinator

    def test_2_registration_gives_student_role_and_hashes_password(self):
        self.register("new.student@test.edu")
        with flask_app.app_context():
            user = User.query.filter_by(email="new.student@test.edu").one()
            self.assertEqual(user.role, "student")
            self.assertNotIn(PASSWORD, user.password_hash)
            self.assertTrue(user.password_hash.startswith(("scrypt:", "pbkdf2:")))
        weak = flask_app.test_client().post("/desk/register", data={
            "name": "Weak", "email": "weak@test.edu", "password": "12345678", "confirm": "12345678"})
        self.assertEqual(weak.status_code, 200)  # rejected: letters and numbers are required

    def test_3_role_permissions_are_enforced_on_the_server(self):
        student = self.register("perm.student@test.edu")
        advisor = self.client_for("advisor@test.edu")
        for path in ("/desk/users", "/desk/categories", "/desk/reports", "/desk/inferences"):
            self.assertEqual(student.get(path).status_code, 403, path)
        self.assertEqual(advisor.get("/desk/users").status_code, 403)
        self.assertEqual(advisor.get("/desk/reports").status_code, 200)
        with flask_app.app_context():
            me = User.query.filter_by(email="perm.student@test.edu").one()
            coordinator = User.query.filter_by(email="coord@test.edu").one()
        # A student cannot promote anyone, including themselves.
        self.assertEqual(student.post(f"/desk/users/{me.id}/role", data={"role": "coordinator"}).status_code, 403)
        # A coordinator cannot change their own role.
        coord = self.client_for("coord@test.edu")
        coord.post(f"/desk/users/{coordinator.id}/role", data={"role": "student"})
        with flask_app.app_context():
            self.assertEqual(db.session.get(User, coordinator.id).role, COORDINATOR)

    def test_4_complete_workflow_with_the_model(self):
        student = self.register("flow.student@test.edu", "Flow Student")
        page = student.get("/desk/inquiries/new").get_data(as_text=True)
        category = re.search(r'<option value="(\d+)"', page).group(1)
        response = student.post("/desk/inquiries/new", data={
            "title": "Minimum group size",
            "question": "What is the minimum number of students required to open an academic group?",
            "category_id": category})
        self.assertEqual(response.status_code, 302)
        with flask_app.app_context():
            inquiry = Inquiry.query.filter_by(title="Minimum group size").one()
            inference = Inference.query.filter_by(inquiry_id=inquiry.id).one()
            self.assertIsNotNone(inference.top_page)
            self.assertIn("random_forest", inference.model_version)
            self.assertIn(inquiry.status, ("new", "escalated"))
            inquiry_id, document, page_number = inquiry.id, inference.top_document, inference.top_page
            escalated = inquiry.status == "escalated"

        # Another student cannot see it.
        other = self.register("other.student@test.edu")
        self.assertEqual(other.get(f"/desk/inquiries/{inquiry_id}").status_code, 403)

        staff = self.client_for("coord@test.edu" if escalated else "advisor@test.edu")
        self.assertEqual(staff.post(f"/desk/inquiries/{inquiry_id}/take").status_code, 302)
        staff.post(f"/desk/inquiries/{inquiry_id}/answer", data={
            "source": f"{document}|{page_number}",
            "answer": "An academic group opens with at least eight students, according to the regulations."})
        with flask_app.app_context():
            inquiry = db.session.get(Inquiry, inquiry_id)
            self.assertEqual(inquiry.status, ANSWERED)
            self.assertTrue(inquiry.model_was_right)

        # Only the owner rates and closes it.
        self.assertEqual(other.post(f"/desk/inquiries/{inquiry_id}/rate", data={"rating": 5}).status_code, 403)
        student.post(f"/desk/inquiries/{inquiry_id}/rate", data={"rating": 5, "comment": "Clear"})
        with flask_app.app_context():
            self.assertEqual(db.session.get(Inquiry, inquiry_id).status, CLOSED)
        report = self.client_for("coord@test.edu").get("/desk/reports").get_data(as_text=True)
        self.assertIn("first page confirmed by the advisor (1 of 1)", report)
        csv_text = self.client_for("coord@test.edu").get("/desk/reports/labeled.csv").get_data(as_text=True)
        self.assertIn("minimum number of students", csv_text)

    def test_5_validation_rules(self):
        student = self.register("valid.student@test.edu")
        response = student.post("/desk/inquiries/new", data={"title": "Hi", "question": "Too short", "category_id": 1})
        self.assertEqual(response.status_code, 200)
        self.assertIn("must have between", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
