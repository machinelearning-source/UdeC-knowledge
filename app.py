import os

import streamlit as st

from core.pipeline import query_rag_system, warm_up

st.set_page_config(page_title="UdeC Knowledge Hub", page_icon="🎓", layout="wide")

if not os.environ.get("GROQ_API_KEY"):
    try:
        for _secret_key, _secret_value in st.secrets.items():
            os.environ.setdefault(_secret_key, str(_secret_value))
    except Exception:
        pass


@st.cache_resource(show_spinner="Loading the search engine (only once)...")
def _warm_engine():
    warm_up()
    return True


GREEN_DARK = "#00482b"
GREEN = "#007b3e"
LIME = "#79c000"
GOLD = "#ffd700"

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "Hello! I am your RAG assistant. What would you like to ask today "
        "about the university regulations or guidelines?"
    ),
}

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

CALENDAR_ACTIVITIES = [
    "Online registration",
    "Publication of shortlisted applicants",
    "Admission test",
    "Publication of admitted students lists",
    "Institutional characterization - new regular students",
    "Institutional characterization - returning, transfer and re-entry students",
    "Payment - new students",
    "Optional students",
    "Ordinary tuition payment - returning students",
    "Publication of class schedules",
    "Academic registration - new regular students",
    "Academic registration - returning students",
    "University life welcome experience",
    "Start of classes",
    "Availability of student lists",
    "Adding of thematic units",
    "Cancellation of thematic units",
    "Semester cancellation",
    "Voluntary withdrawal",
    "CAI - Moment 1 (diagnostic)",
    "CAI - Moment 2, remedial, first attempt (satisfactory)",
    "CAI - Moment 2, remedial, second attempt (satisfactory)",
    "CAI 21st Century Generation course - Moment 1 (diagnostic)",
    "CAI 21st Century Generation course - Moment 2, remedial, first attempt (satisfactory)",
    "CAI 21st Century Generation course - Moment 2, remedial, second attempt (satisfactory)",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), first attempt",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), second attempt",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), third attempt",
    "Deadline for grade entry per term",
    "University week",
    "End of academic period",
    "Closing process and setup of the new academic period",
    "Registration for intersemester courses",
    "Issuance of intersemester course receipts",
    "Payment of intersemester course",
    "Start of intersemester courses",
    "Grade entry for intersemester courses",
    "Document review period - ordinary degrees",
    "Ordinary graduations",
    "Document review period - extraordinary degrees",
    "Extraordinary graduations",
    "Application for transfers and re-entries",
    "Deadline for approval of transfers and re-entries",
    "Deadline for reporting academic changes through SIS",
    "Deadline to register degree option / advanced semester",
    "Deadline for approval of degree option / advanced semester",
    "Student evaluation of teachers",
    "Teacher self-assessment",
    "Coordinators' evaluation of teachers",
    "Faculty vacations (tenured staff)",
    "Student vacations",
    "Holy Week",
    "Online registration (postgraduate)",
    "Selection interviews (postgraduate)",
    "Publication of admitted students (postgraduate)",
    "Institutional characterization - new regular students (postgraduate)",
    "Tuition payment (postgraduate)",
    "Homologation process - degree option / advanced semester (postgraduate)",
    "Academic registration (postgraduate)",
    "University life welcome experience (postgraduate)",
    "Start of classes (postgraduate)",
    "End of academic period (postgraduate)",
    "Closing process and setup of the new academic period (postgraduate)",
    "Document review period - ordinary degrees (postgraduate)",
    "Ordinary graduations (postgraduate)",
    "Document review period - extraordinary degrees (postgraduate)",
    "Extraordinary graduations (postgraduate)",
]

CALENDAR_OFFICIAL_NAMES = {
    "Online registration": "Inscripciones En Línea",
    "Publication of shortlisted applicants": "Publicación Preseleccionados",
    "Admission test": "Prueba De Admisión",
    "Publication of admitted students lists": "Publicación De Listas Admitidos",
    "Institutional characterization - new regular students": "Caracterización Institucional Nuevos Regulares",
    "Institutional characterization - returning, transfer and re-entry students": "Caracterización Institucional Estudiantes Antiguos No Caracterizados - Transferencias - Reingresos",
    "Payment - new students": "Pago Estudiantes Nuevos",
    "Optional students": "Estudiantes Opcionales",
    "Ordinary tuition payment - returning students": "Pago Ordinario De Matricula Antiguos",
    "Publication of class schedules": "Publicación De Horarios",
    "Academic registration - new regular students": "Registro Académico Estudiantes Nuevos Regulares",
    "Academic registration - returning students": "Registro Académico Estudiantes Antiguos",
    "University life welcome experience": "Experiencia De Acogida A La Vida Universitaria",
    "Start of classes": "Inicio Clases",
    "Availability of student lists": "Disponibilidad De Listados De Estudiantes",
    "Adding of thematic units": "Adición De Núcleos Temáticos",
    "Cancellation of thematic units": "Cancelación De Núcleos Temáticos",
    "Semester cancellation": "Cancelación De Semestre",
    "Voluntary withdrawal": "Cancelación Retiro Voluntario",
    "CAI - Moment 1 (diagnostic)": "Presentación Campos De Aprendizaje Institucional CAI- Momento 1 (Diagnóstico)",
    "CAI - Moment 2, remedial, first attempt (satisfactory)": "Presentación Campos De Aprendizaje Institucional CAI- Momento 2 Nivelatorio Primera Vez (Satisfactorio)",
    "CAI - Moment 2, remedial, second attempt (satisfactory)": "Presentación Campos De Aprendizaje Institucional CAI- Momento 2 Nivelatorio Segunda Vez (Satisfactorio)",
    "CAI 21st Century Generation course - Moment 1 (diagnostic)": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 1 (Diagnóstico)",
    "CAI 21st Century Generation course - Moment 2, remedial, first attempt (satisfactory)": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 2 -Nivelatorio Primera Vez (Satisfactorio)",
    "CAI 21st Century Generation course - Moment 2, remedial, second attempt (satisfactory)": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 2 -Nivelatorio Segunda Vez (Satisfactorio)",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), first attempt": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) -- Primera Vez",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), second attempt": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) - Segunda Vez",
    "CAI 21st Century Generation course - Moment 3, on-site (advanced level), third attempt": "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) - Tercera Vez",
    "Deadline for grade entry per term": "Fecha Limite Para Registro De Notas Por Corte",
    "University week": "Semana Universitaria",
    "End of academic period": "Finalización De Periodo Académico",
    "Closing process and setup of the new academic period": "Proceso Cierre Y Parametrización Nuevo Periodo Académico",
    "Registration for intersemester courses": "Inscripciones Cursos Intersemestrales",
    "Issuance of intersemester course receipts": "Expedición Recibos Cursos Intersemestrales",
    "Payment of intersemester course": "Pago Curso Intersemestral",
    "Start of intersemester courses": "Inicio Cursos Intersemestrales",
    "Grade entry for intersemester courses": "Registro De Notas Intersemestrales",
    "Document review period - ordinary degrees": "Periodo Revisión Documentos Para Grado Ordinarios",
    "Ordinary graduations": "Grados Ordinarios",
    "Document review period - extraordinary degrees": "Periodo Revisión Documentos Para Grado Extraordinarios",
    "Extraordinary graduations": "Grados Extraordinarios",
    "Application for transfers and re-entries": "Solicitud De Traslados, Transferencias Y Reingresos",
    "Deadline for approval of transfers and re-entries": "Fecha Limite De Aprobación Traslados, Transferencias Y Reingresos",
    "Deadline for reporting academic changes through SIS": "Fecha Máxima De Reporte De Novedades Académicas A Traves Del SIS (Traslados, Transferencias Y Reingresos)",
    "Deadline to register degree option / advanced semester": "Fecha Máxima Inscripción Opción De Trabajo De Grado/ Semestre Avanzado",
    "Deadline for approval of degree option / advanced semester": "Fecha Máxima De Aprobación Opción De Grado/ Semestre Avanzado",
    "Student evaluation of teachers": "Evaluación De Los Estudiantes A Los Docentes",
    "Teacher self-assessment": "Autoevaluación Docentes",
    "Coordinators' evaluation of teachers": "Evaluación De Los Coordinadores A Los Docentes",
    "Faculty vacations (tenured staff)": "Vacaciones Docentes Planta",
    "Student vacations": "Vacaciones Estudiantes",
    "Holy Week": "Semana Santa",
    "Online registration (postgraduate)": "Inscripciones En Línea (Posgrados)",
    "Selection interviews (postgraduate)": "Entrevistas De Selección (Posgrados)",
    "Publication of admitted students (postgraduate)": "Publicación Admitidos (Posgrados)",
    "Institutional characterization - new regular students (postgraduate)": "Caracterización Institucional Estudiantes Nuevo Regular - Posgrados (Posgrados)",
    "Tuition payment (postgraduate)": "Pago De Matricula (Posgrados)",
    "Homologation process - degree option / advanced semester (postgraduate)": "Porceso De Homologación (Opción De Titulación-Semestre Avanzado) (Posgrados)",
    "Academic registration (postgraduate)": "Registro De Matricula Académica (Posgrados)",
    "University life welcome experience (postgraduate)": "Experiencia De Acogida A La Vida Universtaria (Posgrados)",
    "Start of classes (postgraduate)": "Inicio Clases (Posgrados)",
    "End of academic period (postgraduate)": "Finalización De Periodo Académico (Posgrados)",
    "Closing process and setup of the new academic period (postgraduate)": "Proceso Cierre Y Parametrización Nuevo Periodo Académico (Posgrados)",
    "Document review period - ordinary degrees (postgraduate)": "Periodo Revisión Documentos Para Grados Ordinarios (Posgrados)",
    "Ordinary graduations (postgraduate)": "Grados Ordinarios (Posgrados)",
    "Document review period - extraordinary degrees (postgraduate)": "Periodo Revisión Documentos Para Grados Extraordinarios (Posgrados)",
    "Extraordinary graduations (postgraduate)": "Grados Extraordinarios (Posgrados)",
}

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
CALENDAR_RETRIEVAL_QUESTION = (
    "According to the 2026 academic calendar, when does the following activity take "
    "place: {option} (official name in the document: {official})? Give the dates and the month. "
    "Answer in English without repeating the Spanish official name."
)

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    [data-testid="stToolbar"] {{visibility: hidden;}}
    [data-testid="stDecoration"] {{display: none;}}
    footer {{visibility: hidden;}}
    [data-testid="stHeader"] {{background: transparent;}}

    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {{
        background-color: #edf7e7 !important;
        background-image: none !important;
    }}
    [data-testid="stSidebar"] {{
        border-right: 4px solid {GOLD};
    }}
    [data-testid="stSidebar"] * {{color: {GREEN_DARK};}}
    [data-testid="stSidebar"] h4 {{color: {GREEN_DARK}; font-weight: 800;}}
    [data-testid="stSidebar"] .stButton > button {{
        background: {GREEN_DARK} !important;
        border: 2px solid {GREEN_DARK} !important;
        border-radius: 12px;
        width: 100%;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(0, 72, 43, 0.22);
    }}
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button * {{
        color: {GOLD} !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {GOLD} !important;
        border-color: {GOLD} !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover * {{
        color: {GREEN_DARK} !important;
    }}
    .udec-brand {{
        background: rgba(0, 72, 43, 0.06);
        border-left: 5px solid {GOLD};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 18px;
    }}
    .udec-brand h2 {{margin: 0; font-size: 1.35rem; color: {GREEN_DARK};}}
    .udec-brand h2 span {{color: #8a6d00;}}
    .udec-brand p {{margin: 4px 0 0; font-size: 0.82rem; color: #3c5a49;}}

    .stApp {{
        background:
            radial-gradient(circle at 8% 0%, rgba(121, 192, 0, 0.38) 0%, rgba(121, 192, 0, 0) 45%),
            radial-gradient(circle at 96% 8%, rgba(255, 215, 0, 0.45) 0%, rgba(255, 215, 0, 0) 40%),
            linear-gradient(160deg, #dcefd4 0%, #f3fbee 45%, #faf1cc 100%) !important;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{background: transparent !important;}}
    .block-container {{padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1080px;}}

    .udec-hero {{
        background-color: {GREEN};
        background-image: linear-gradient(100deg, {GREEN_DARK} 0%, {GREEN} 58%, {LIME} 100%);
        border-bottom: 7px solid {GOLD};
        border-radius: 20px;
        padding: 30px 34px;
        color: #ffffff;
        box-shadow: 0 12px 28px rgba(0, 72, 43, 0.30);
        margin-bottom: 6px;
    }}
    .udec-hero h1 {{color: #ffffff; font-size: 2.4rem; margin: 0;}}
    .udec-hero h1 span {{color: {GOLD};}}
    .udec-hero p {{margin: 8px 0 0; color: #e7f5ea; font-size: 1rem;}}

    .udec-tagline {{
        color: {GREEN_DARK}; font-weight: 600; font-size: 0.95rem; margin: 10px 0 18px;
    }}

    .udec-card-head {{
        border-radius: 12px;
        padding: 10px 14px;
        color: #ffffff;
        margin-bottom: 6px;
    }}
    .udec-card-head .title {{font-weight: 700; font-size: 0.98rem; display: block;}}
    .udec-card-head .count {{font-size: 0.85rem; opacity: 1;}}
    .udec-card-green {{background: linear-gradient(120deg, {GREEN_DARK}, {GREEN});}}
    .udec-card-gold {{background: linear-gradient(120deg, #b8860b, {GOLD}); color: {GREEN_DARK};}}
    .udec-card-gold .title {{color: {GREEN_DARK};}}
    .udec-card-lime {{background: linear-gradient(120deg, {GREEN}, {LIME});}}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 16px;
        border: 1px solid rgba(0, 72, 43, 0.30);
        background: linear-gradient(180deg, #e8f6e2 0%, #f4fcef 100%);
        box-shadow: 0 8px 22px rgba(0, 72, 43, 0.12);
    }}

    div.stButton > button {{
        border: 2px solid {GREEN};
        border-radius: 12px;
        background: #ffffff;
        color: {GREEN_DARK};
        font-weight: 600;
        transition: all 0.15s ease;
    }}
    div.stButton > button:hover {{
        background: {GREEN};
        border-color: {GREEN};
        color: #ffffff;
    }}
    div.stButton > button[kind="primary"] {{
        background: {GOLD};
        border-color: {GOLD};
        color: {GREEN_DARK};
    }}
    div.stButton > button[kind="primary"]:hover {{
        background: {LIME};
        border-color: {LIME};
        color: #ffffff;
    }}
    [data-testid="stChatMessage"] {{
        background: #f5fcf1;
        border: 1px solid rgba(0, 72, 43, 0.14);
        border-left: 6px solid {LIME};
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0, 72, 43, 0.10);
        padding: 4px 10px;
        margin-bottom: 8px;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        border-left-color: {GREEN};
        background: #e6f6df;
    }}
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {{
        border: 2px solid {GREEN};
        border-radius: 12px;
    }}
    [data-testid="stExpander"] summary {{color: {GREEN}; font-weight: 600;}}
    .udec-footer {{
        margin-top: 26px;
        padding-top: 10px;
        border-top: 3px solid {GOLD};
        color: {GREEN_DARK};
        font-size: 0.8rem;
        font-weight: 700;
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="udec-brand">
            <h2>UdeC-<span>Knowledge</span></h2>
            <p>Intelligent Knowledge Management System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "show_about" not in st.session_state:
        st.session_state.show_about = False

    st.markdown("#### Quick actions")
    if st.button("🔄 Reload page", use_container_width=True):
        st.session_state.clear()
        st.query_params["reloaded"] = "1"
        st.rerun()

    if st.button("🧹 New conversation", use_container_width=True):
        st.session_state.messages = [dict(WELCOME_MESSAGE)]
        st.session_state.pending_question = None
        st.session_state.pending_query = None
        st.session_state.flash = "New conversation started."
        st.rerun()

    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.show_about = not st.session_state.show_about

    if st.session_state.show_about:
        st.markdown(
            "**About this app**  \n"
            "This pilot answers questions about the university's official documents with a "
            "Retrieval-Augmented Generation pipeline:\n\n"
            "1. **Ingestion:** the PDFs are read and split into chunks.\n"
            "2. **Embeddings:** FastEmbed (multilingual MiniLM) runs locally.\n"
            "3. **Retrieval:** FAISS + MMR finds the most similar chunks.\n"
            "4. **Generation:** the Groq model `openai/gpt-oss-120b` writes the answer using only "
            "those chunks, in the language of the question.\n\n"
            "Every answer cites the document and page it came from and says clearly when the "
            "information is not in the indexed documents."
        )

    st.markdown("#### Knowledge base")
    st.markdown(
        """
        - 📄 **4** institutional documents
        - 🔎 **290** searchable sections
        - 🗓️ **2026** academic calendar
        - 🧠 Multilingual local embeddings
        """
    )

st.markdown(
    """
    <div class="udec-hero">
        <h1>🎓 UdeC-<span>Knowledge</span></h1>
        <p>Intelligent Knowledge Management System</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="udec-tagline">Pilot Project: smart search of institutional regulations and university guidelines.</p>',
    unsafe_allow_html=True,
)

flash_message = st.session_state.pop("flash", None)
if flash_message:
    st.success(flash_message)

if st.query_params.get("reloaded") == "1" and "_reload_notice" not in st.session_state:
    st.session_state["_reload_notice"] = True
    st.success("Application reloaded.")

if "messages" not in st.session_state:
    st.session_state.messages = [dict(WELCOME_MESSAGE)]

explore_column, course_column, calendar_column = st.columns(3, gap="medium")

with explore_column:
    with st.container(border=True):
        st.markdown(
            '<div class="udec-card-head udec-card-green"><span class="title">🎓 Student categories</span>'
            '<span class="count">50 student situations</span></div>',
            unsafe_allow_html=True,
        )
        chosen_category = st.selectbox(
            "Choose a student category",
            STUDENT_CATEGORIES,
            key="chosen_category",
            label_visibility="collapsed",
        )
        if st.button("Ask about this category", key="ask_category", type="primary", use_container_width=True):
            st.session_state.pending_question = CATEGORY_QUESTION.format(option=chosen_category)
            st.rerun()

with course_column:
    with st.container(border=True):
        st.markdown(
            '<div class="udec-card-head udec-card-gold"><span class="title">📚 Course homologation</span>'
            '<span class="count">30 courses</span></div>',
            unsafe_allow_html=True,
        )
        chosen_course = st.selectbox(
            "Choose a course",
            HOMOLOGATION_COURSES,
            key="chosen_course",
            label_visibility="collapsed",
        )
        if st.button("Ask about this course", key="ask_course", type="primary", use_container_width=True):
            st.session_state.pending_question = COURSE_QUESTION.format(option=chosen_course)
            st.rerun()

with calendar_column:
    with st.container(border=True):
        st.markdown(
            '<div class="udec-card-head udec-card-lime"><span class="title">🗓️ Academic calendar</span>'
            '<span class="count">67 activities of 2026</span></div>',
            unsafe_allow_html=True,
        )
        chosen_activity = st.selectbox(
            "Choose an academic calendar activity",
            CALENDAR_ACTIVITIES,
            key="chosen_activity",
            label_visibility="collapsed",
        )
        if st.button("Ask about this date", key="ask_calendar", type="primary", use_container_width=True):
            official_name = CALENDAR_OFFICIAL_NAMES[chosen_activity]
            st.session_state.pending_question = CALENDAR_QUESTION.format(option=chosen_activity)
            st.session_state.pending_query = CALENDAR_RETRIEVAL_QUESTION.format(
                option=chosen_activity,
                official=official_name,
            )
            st.rerun()

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Document sources"):
                for source in message["sources"]:
                    st.markdown(source)

typed_prompt = st.chat_input("Type your question here...")
prompt = st.session_state.pop("pending_question", None) or typed_prompt
retrieval_query = st.session_state.pop("pending_query", None) or prompt

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching the documents..."):
        _warm_engine()
        assistant_response, sources = query_rag_system(retrieval_query)

    shown_sources = [] if assistant_response.startswith("Error:") else sources

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
        if shown_sources:
            with st.expander("Document sources"):
                for source in shown_sources:
                    st.markdown(source)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response, "sources": shown_sources}
    )

st.markdown(
    '<div class="udec-footer">FastEmbed + FAISS + Groq · UI v1.8</div>',
    unsafe_allow_html=True,
)
