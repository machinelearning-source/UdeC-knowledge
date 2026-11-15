"""Quick questions of KnowBot: student situations, homologation courses and 2026 calendar activities.

The same menus as the KnowBot chatbot, so the demo can be used the same way: choose an
option, then confirm it.
"""

from .calendar_names import CALENDAR_OFFICIAL_NAMES

STUDENT_CATEGORIES = [
    "New incoming student",
    "Returning student",
    "Transfer student from another university",
    "Student moved between programs of the university",
    "Incoming exchange student",
    "Outgoing exchange student",
    "Student enrolled in two programs at the same time",
    "Student on academic probation",
    "Student with a cumulative average below 3.2",
    "Student who failed the same CAI three times",
    "Student who failed the same CADI three times",
    "Student requesting semester cancellation",
    "Student requesting voluntary withdrawal",
    "Student who did not renew enrollment on time",
    "Student with unpaid tuition fees",
    "Student requesting a grade review",
    "Student accused of academic fraud",
    "Student under a disciplinary sanction",
    "Student with cancelled enrollment",
    "Student about to graduate",
    "Student applying for an extraordinary degree",
    "Student registering the degree option",
    "Student registering the advanced semester option",
    "Student enrolling in an intersemester course",
    "Student requesting course homologation",
    "Student requesting course validation",
    "Student repeating a course",
    "Student in the institutional learning assessment (CAI)",
    "Student requesting a special examination",
    "Student with a disability requiring accommodations",
    "High-performance athlete student",
    "Student artist or cultural representative",
    "Working student",
    "Pregnant student",
    "Student on maternity or paternity leave",
    "Student with a medical incapacity",
    "Student in a vulnerable economic situation",
    "Scholarship holder",
    "Student with an educational loan",
    "Student with academic distinction or honors",
    "Student monitor or teaching assistant",
    "Student member of a research seedbed",
    "Student representative in university governance",
    "Student delegate of the student council",
    "Postgraduate student",
    "Student of a technical or technological program",
    "Student in the distance or virtual modality",
    "Student of a rural campus",
    "Student in the final semester of the program",
    "Student requesting certificates or academic records",
]

HOMOLOGATION_COURSES = [
    "Differential Calculus",
    "Integral Calculus",
    "Multivariable Calculus",
    "Linear Algebra",
    "Differential Equations",
    "Physics: Mechanics",
    "Physics: Electricity and Magnetism",
    "General Chemistry",
    "Descriptive Statistics",
    "Probability and Statistics",
    "Programming I",
    "Programming II",
    "Data Structures",
    "Databases",
    "Computer Networks",
    "Operating Systems",
    "Software Engineering",
    "Artificial Intelligence",
    "Machine Learning",
    "General Accounting",
    "Microeconomics",
    "Macroeconomics",
    "Business Administration",
    "Marketing",
    "Financial Mathematics",
    "English I",
    "English II",
    "Professional Ethics",
    "Research Methodology",
    "Degree Project",
]

CALENDAR_ACTIVITIES = list(CALENDAR_OFFICIAL_NAMES)


CATEGORY_QUESTION = (
    "According to the student regulations, what does the university establish for a "
    "student in this situation: {option}? Mention the requirements, rights and consequences."
)
COURSE_QUESTION = (
    "According to the student regulations, what is the procedure and what are the "
    "requirements to homologate the course {option}?"
)
CALENDAR_QUESTION = (
    "According to the 2026 academic calendar, when does the following activity take "
    "place: {option}? Give the dates and the month."
)
