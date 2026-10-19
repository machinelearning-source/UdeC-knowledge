import os

import streamlit as st

from core.pipeline import query_rag_system, warm_up

st.set_page_config(page_title="UdeC Knowledge Hub", page_icon="🎓", layout="centered")

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
    "New incoming student (nuevo ingreso)",
    "Returning student (reingreso)",
    "Transfer student from another university (transferencia)",
    "Student moved between programs of the university (traslado)",
    "Incoming exchange student (movilidad entrante)",
    "Outgoing exchange student (movilidad saliente)",
    "Student enrolled in two programs at the same time (simultaneidad de programa)",
    "Student on academic probation (prueba académica)",
    "Student with a low cumulative average (promedio inferior a 3.2)",
    "Student who failed the same CAI three times",
    "Student who failed the same CADI three times",
    "Student requesting cancellation of the semester (cancelación de semestre)",
    "Student requesting voluntary withdrawal (retiro voluntario)",
    "Student who did not renew enrollment on time (renovación de matrícula)",
    "Student with unpaid tuition fees (derechos pecuniarios)",
    "Student requesting a grade review (revisión de calificaciones)",
    "Student accused of academic fraud (fraude académico)",
    "Student under a disciplinary sanction (sanción disciplinaria)",
    "Student with cancelled enrollment (cancelación de matrícula)",
    "Student about to graduate (grados ordinarios)",
    "Student applying for an extraordinary degree (grados extraordinarios)",
    "Student registering the degree option (opción de trabajo de grado)",
    "Student registering the advanced semester option (semestre avanzado)",
    "Student enrolling in an intersemester course (curso intersemestral)",
    "Student requesting course homologation (homologación)",
    "Student requesting validation of courses (validación)",
    "Student repeating a course (repetición de asignatura)",
    "Student in the institutional learning assessment (CAI)",
    "Student requesting a special examination (examen especial)",
    "Student with a disability requiring accommodations",
    "High-performance athlete student (deportista)",
    "Student artist or cultural representative (artista)",
    "Working student (estudiante trabajador)",
    "Pregnant student (embarazo)",
    "Student on maternity or paternity leave (licencia de maternidad o paternidad)",
    "Student with a medical incapacity (incapacidad médica)",
    "Student in a vulnerable economic situation (vulnerabilidad económica)",
    "Scholarship holder (becario)",
    "Student with an educational loan (crédito educativo)",
    "Student with academic distinction or honors (distinción académica)",
    "Student monitor or teaching assistant (monitor)",
    "Student member of a research seedbed (semillero de investigación)",
    "Student representative in university governance (representante estudiantil)",
    "Student delegate of the student council (consejo estudiantil)",
    "Postgraduate student (posgrado)",
    "Student of a technical or technological program (técnico o tecnológico)",
    "Student in the distance or virtual modality (modalidad a distancia)",
    "Student of a rural campus (sede rural)",
    "Student in the final semester of the program (último semestre)",
    "Student requesting certificates or academic constancias",
]

HOMOLOGATION_COURSES = [
    "Cálculo Diferencial",
    "Cálculo Integral",
    "Cálculo Multivariado",
    "Álgebra Lineal",
    "Ecuaciones Diferenciales",
    "Física Mecánica",
    "Física Electricidad y Magnetismo",
    "Química General",
    "Estadística Descriptiva",
    "Probabilidad y Estadística",
    "Programación I",
    "Programación II",
    "Estructuras de Datos",
    "Bases de Datos",
    "Redes de Computadores",
    "Sistemas Operativos",
    "Ingeniería de Software",
    "Inteligencia Artificial",
    "Machine Learning",
    "Contabilidad General",
    "Microeconomía",
    "Macroeconomía",
    "Administración de Empresas",
    "Mercadeo",
    "Matemáticas Financieras",
    "Inglés I",
    "Inglés II",
    "Ética Profesional",
    "Metodología de la Investigación",
    "Trabajo de Grado",
]

CALENDAR_ACTIVITIES = [
    "Inscripciones en línea",
    "Publicación de preseleccionados",
    "Prueba de admisión",
    "Publicación de listas de admitidos",
    "Caracterización institucional estudiantes nuevos regulares",
    "Caracterización institucional estudiantes antiguos no caracterizados",
    "Caracterización de transferencias y reingresos",
    "Pago estudiantes nuevos",
    "Pago estudiantes opcionales",
    "Pago ordinario de matrícula estudiantes antiguos",
    "Publicación de horarios",
    "Registro académico estudiantes nuevos regulares",
    "Registro académico estudiantes antiguos",
    "Experiencia de acogida a la vida universitaria",
    "Inicio de clases",
    "Disponibilidad de listados de estudiantes",
    "Adición de núcleos temáticos",
    "Cancelación de núcleos temáticos",
    "Cancelación de semestre",
    "Cancelación por retiro voluntario",
    "Presentación CAI momento 1 (diagnóstico)",
    "Presentación CAI momento 2 nivelatorio primera vez",
    "Presentación CAI momento 2 nivelatorio segunda vez",
    "Cátedra Generación Siglo 21 momento 3 avanzado",
    "Fecha límite para registro de notas por corte",
    "Semana universitaria",
    "Finalización de periodo académico",
    "Proceso de cierre y parametrización del nuevo periodo académico",
    "Inscripciones cursos intersemestrales",
    "Expedición de recibos cursos intersemestrales",
    "Pago curso intersemestral",
    "Inicio cursos intersemestrales",
    "Registro de notas intersemestrales",
    "Periodo de revisión de documentos para grado ordinario",
    "Grados ordinarios",
    "Periodo de revisión de documentos para grado extraordinario",
    "Grados extraordinarios",
    "Solicitud de traslados, transferencias y reingresos",
    "Fecha límite de aprobación de traslados, transferencias y reingresos",
    "Fecha máxima de inscripción de opción de trabajo de grado o semestre avanzado",
    "Fecha máxima de aprobación de opción de grado o semestre avanzado",
    "Evaluación de los estudiantes a los docentes",
    "Autoevaluación de docentes",
    "Evaluación de los coordinadores a los docentes",
    "Vacaciones docentes de planta",
    "Vacaciones estudiantes",
    "Semana Santa",
    "Inscripciones en línea de posgrado",
    "Entrevistas de selección de posgrado",
    "Publicación de admitidos en posgrado",
    "Pago de matrícula de posgrado",
    "Proceso de homologación de posgrado (opción de titulación o semestre avanzado)",
    "Registro de matrícula académica de posgrado",
]

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

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    [data-testid="stToolbar"] {{visibility: hidden;}}
    [data-testid="stDecoration"] {{display: none;}}
    footer {{visibility: hidden;}}
    [data-testid="stHeader"] {{background: transparent;}}
    .stApp {{
        background:
            radial-gradient(circle at 12% 0%, rgba(121, 192, 0, 0.35) 0%, rgba(121, 192, 0, 0) 45%),
            radial-gradient(circle at 92% 8%, rgba(255, 215, 0, 0.38) 0%, rgba(255, 215, 0, 0) 40%),
            linear-gradient(160deg, #e9fbe4 0%, #f6fff0 40%, #fff9e0 100%) !important;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{background: transparent !important;}}
    .block-container {{padding-top: 2rem; max-width: 850px;}}
    [data-testid="stChatMessage"] {{
        background: #ffffff;
        border: 1px solid rgba(0, 72, 43, 0.12);
        border-left: 6px solid {LIME};
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0, 72, 43, 0.10);
        padding: 4px 10px;
        margin-bottom: 8px;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        border-left-color: {GREEN};
        background: #f4fdf6;
    }}
    .udec-hero {{
        background: linear-gradient(100deg, {GREEN_DARK} 0%, {GREEN} 62%, {LIME} 100%);
        border-bottom: 7px solid {GOLD};
        border-radius: 18px;
        padding: 26px 30px;
        color: #ffffff;
        box-shadow: 0 10px 24px rgba(0, 72, 43, 0.28);
        margin-bottom: 14px;
    }}
    .udec-hero h1 {{color: #ffffff; font-size: 2.1rem; margin: 0;}}
    .udec-hero h1 span {{color: {GOLD};}}
    .udec-hero p {{margin: 8px 0 0; color: #e7f5ea; font-size: 0.95rem;}}
    .udec-tagline {{color: {GREEN_DARK}; font-weight: 600; margin: 0 0 10px;}}
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
    [data-testid="stExpander"] summary {{color: {GREEN}; font-weight: 600;}}
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {{
        border: 2px solid {GREEN};
        border-radius: 12px;
    }}
    .udec-footer {{
        margin-top: 28px;
        padding-top: 10px;
        border-top: 3px solid {GOLD};
        color: {GREEN};
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
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

if "messages" not in st.session_state:
    st.session_state.messages = [dict(WELCOME_MESSAGE)]
if "show_about" not in st.session_state:
    st.session_state.show_about = False

reload_col, new_chat_col, about_col = st.columns(3)

if reload_col.button("🔄 Reload page", use_container_width=True):
    st.rerun()

if new_chat_col.button("🧹 New conversation", use_container_width=True):
    st.session_state.messages = [dict(WELCOME_MESSAGE)]
    st.session_state.pending_question = None
    st.session_state.show_about = False
    st.rerun()

if about_col.button("ℹ️ About", use_container_width=True):
    st.session_state.show_about = not st.session_state.show_about

if st.session_state.show_about:
    st.info(
        "**About this app** — This pilot answers questions about the university's official "
        "documents using a Retrieval-Augmented Generation pipeline:\n\n"
        "1. **Ingestion:** the PDF files in `data/documents` are read and split into chunks.\n"
        "2. **Embeddings:** each chunk is converted into a vector with the multilingual model "
        "`paraphrase-multilingual-MiniLM-L12-v2` (FastEmbed, runs locally).\n"
        "3. **Retrieval:** FAISS finds the most similar chunks for your question.\n"
        "4. **Generation:** the Groq LLM (`openai/gpt-oss-120b`) writes the answer using only those "
        "chunks, in the language of your question.\n\n"
        "Every answer cites the document and page it came from, and the assistant states clearly "
        "when the answer is not present in the indexed documents."
    )

st.caption("Quick questions")

categories_tab, homologation_tab, calendar_tab = st.tabs(
    ["🎓 Student categories", "📚 Course homologation", "🗓️ Academic calendar"]
)

with categories_tab:
    chosen_category = st.selectbox(
        "Choose a student category",
        STUDENT_CATEGORIES,
        key="chosen_category",
        help="50 student situations taken from the student regulations.",
    )
    if st.button("Ask about this category", key="ask_category", type="primary", use_container_width=True):
        st.session_state.pending_question = CATEGORY_QUESTION.format(option=chosen_category)
        st.rerun()

with homologation_tab:
    chosen_course = st.selectbox(
        "Choose a course",
        HOMOLOGATION_COURSES,
        key="chosen_course",
        help="30 courses that can be submitted for a homologation study.",
    )
    if st.button("Ask about this course", key="ask_course", type="primary", use_container_width=True):
        st.session_state.pending_question = COURSE_QUESTION.format(option=chosen_course)
        st.rerun()

with calendar_tab:
    chosen_activity = st.selectbox(
        "Choose an academic calendar activity",
        CALENDAR_ACTIVITIES,
        key="chosen_activity",
        help="Activities and dates of the official 2026 academic calendar.",
    )
    if st.button("Ask about this date", key="ask_calendar", type="primary", use_container_width=True):
        st.session_state.pending_question = CALENDAR_QUESTION.format(option=chosen_activity)
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

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching the documents..."):
        _warm_engine()
        assistant_response, sources = query_rag_system(prompt)

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
    '<div class="udec-footer">FastEmbed + FAISS + Groq</div>',
    unsafe_allow_html=True,
)
