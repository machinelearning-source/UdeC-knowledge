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
    "Inscripciones En Línea",
    "Publicación Preseleccionados",
    "Prueba De Admisión",
    "Publicación De Listas Admitidos",
    "Caracterización Institucional Nuevos Regulares",
    "Caracterización Institucional Estudiantes Antiguos No Caracterizados - Transferencias - Reingresos",
    "Pago Estudiantes Nuevos",
    "Estudiantes Opcionales",
    "Pago Ordinario De Matricula Antiguos",
    "Publicación De Horarios",
    "Registro Académico Estudiantes Nuevos Regulares",
    "Registro Académico Estudiantes Antiguos",
    "Experiencia De Acogida A La Vida Universitaria",
    "Inicio Clases",
    "Disponibilidad De Listados De Estudiantes",
    "Adición De Núcleos Temáticos",
    "Cancelación De Núcleos Temáticos",
    "Cancelación De Semestre",
    "Cancelación Retiro Voluntario",
    "Presentación Campos De Aprendizaje Institucional CAI- Momento 1 (Diagnóstico)",
    "Presentación Campos De Aprendizaje Institucional CAI- Momento 2 Nivelatorio Primera Vez (Satisfactorio)",
    "Presentación Campos De Aprendizaje Institucional CAI- Momento 2 Nivelatorio Segunda Vez (Satisfactorio)",
    "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 1 (Diagnóstico)",
    "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 2 -Nivelatorio Primera Vez (Satisfactorio)",
    "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 2 -Nivelatorio Segunda Vez (Satisfactorio)",
    "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) -- Primera Vez",
    "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) - Segunda Vez",
    "Presentación Campo De Aprendizaje Institucional CAI Cátedra Generación Siglo 21 Momento 3 - Presencial (Nivel Avanzado) - Tercera Vez",
    "Fecha Limite Para Registro De Notas Por Corte",
    "Semana Universitaria",
    "Finalización De Periodo Académico",
    "Proceso Cierre Y Parametrización Nuevo Periodo Académico",
    "Inscripciones Cursos Intersemestrales",
    "Expedición Recibos Cursos Intersemestrales",
    "Pago Curso Intersemestral",
    "Inicio Cursos Intersemestrales",
    "Registro De Notas Intersemestrales",
    "Periodo Revisión Documentos Para Grado Ordinarios",
    "Grados Ordinarios",
    "Periodo Revisión Documentos Para Grado Extraordinarios",
    "Grados Extraordinarios",
    "Solicitud De Traslados, Transferencias Y Reingresos",
    "Fecha Limite De Aprobación Traslados, Transferencias Y Reingresos",
    "Fecha Máxima De Reporte De Novedades Académicas A Traves Del SIS (Traslados, Transferencias Y Reingresos)",
    "Fecha Máxima Inscripción Opción De Trabajo De Grado/ Semestre Avanzado",
    "Fecha Máxima De Aprobación Opción De Grado/ Semestre Avanzado",
    "Evaluación De Los Estudiantes A Los Docentes",
    "Autoevaluación Docentes",
    "Evaluación De Los Coordinadores A Los Docentes",
    "Vacaciones Docentes Planta",
    "Vacaciones Estudiantes",
    "Semana Santa",
    "Inscripciones En Línea (Posgrados)",
    "Entrevistas De Selección (Posgrados)",
    "Publicación Admitidos (Posgrados)",
    "Caracterización Institucional Estudiantes Nuevo Regular - Posgrados (Posgrados)",
    "Pago De Matricula (Posgrados)",
    "Porceso De Homologación (Opción De Titulación-Semestre Avanzado) (Posgrados)",
    "Registro De Matricula Académica (Posgrados)",
    "Experiencia De Acogida A La Vida Universtaria (Posgrados)",
    "Inicio Clases (Posgrados)",
    "Finalización De Periodo Académico (Posgrados)",
    "Proceso Cierre Y Parametrización Nuevo Periodo Académico (Posgrados)",
    "Periodo Revisión Documentos Para Grados Ordinarios (Posgrados)",
    "Grados Ordinarios (Posgrados)",
    "Periodo Revisión Documentos Para Grados Extraordinarios (Posgrados)",
    "Grados Extraordinarios (Posgrados)",
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

    [data-testid="stSidebar"] {{
        background: linear-gradient(175deg, {GREEN_DARK} 0%, {GREEN} 70%, {LIME} 160%) !important;
    }}
    [data-testid="stSidebar"] * {{color: #ffffff;}}
    [data-testid="stSidebarContent"] {{
        background: linear-gradient(175deg, {GREEN_DARK} 0%, {GREEN} 70%, {LIME} 160%);
    }}
    .udec-brand {{
        background: rgba(255, 255, 255, 0.10);
        border-left: 5px solid {GOLD};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 18px;
    }}
    .udec-brand h2 {{margin: 0; font-size: 1.35rem; color: #ffffff;}}
    .udec-brand h2 span {{color: {GOLD};}}
    .udec-brand p {{margin: 4px 0 0; font-size: 0.82rem; color: #d8f0dc;}}

    .stApp {{
        background:
            radial-gradient(circle at 10% 0%, rgba(121, 192, 0, 0.30) 0%, rgba(121, 192, 0, 0) 42%),
            radial-gradient(circle at 94% 6%, rgba(255, 215, 0, 0.34) 0%, rgba(255, 215, 0, 0) 38%),
            linear-gradient(160deg, #e9fbe4 0%, #f6fff0 42%, #fff9e0 100%) !important;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{background: transparent !important;}}
    .block-container {{padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1080px;}}

    .udec-hero {{
        background: linear-gradient(100deg, {GREEN_DARK} 0%, {GREEN} 58%, {LIME} 100%);
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
    .udec-card-head .count {{font-size: 0.75rem; opacity: 0.92;}}
    .udec-card-green {{background: linear-gradient(120deg, {GREEN_DARK}, {GREEN});}}
    .udec-card-gold {{background: linear-gradient(120deg, #b8860b, {GOLD}); color: {GREEN_DARK};}}
    .udec-card-gold .title {{color: {GREEN_DARK};}}
    .udec-card-lime {{background: linear-gradient(120deg, {GREEN}, {LIME});}}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 16px;
        border-color: rgba(0, 72, 43, 0.22);
        background: rgba(255, 255, 255, 0.78);
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
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {{
        border: 2px solid {GREEN};
        border-radius: 12px;
    }}
    [data-testid="stExpander"] summary {{color: {GREEN}; font-weight: 600;}}
    .udec-footer {{
        margin-top: 26px;
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
        st.rerun()

    if st.button("🧹 New conversation", use_container_width=True):
        st.session_state.messages = [dict(WELCOME_MESSAGE)]
        st.session_state.pending_question = None
        st.session_state.show_about = False
        st.rerun()

    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.show_about = not st.session_state.show_about

    st.markdown("#### Knowledge base")
    st.markdown(
        """
        - 📄 **4** institutional documents
        - 🔎 **290** searchable sections
        - 🗓️ **2026** academic calendar
        - 🧠 Multilingual local embeddings
        """
    )

if st.session_state.show_about:
    st.info(
        "**About this app** — This pilot answers questions about the university's official "
        "documents using a Retrieval-Augmented Generation pipeline:\n\n"
        "1. **Ingestion:** the PDF files in `data/documents` are read and split into chunks.\n"
        "2. **Embeddings:** each chunk is converted into a vector with the multilingual model "
        "`paraphrase-multilingual-MiniLM-L12-v2` (FastEmbed, runs locally).\n"
        "3. **Retrieval:** FAISS finds the most similar chunks for your question (MMR).\n"
        "4. **Generation:** the Groq LLM (`openai/gpt-oss-120b`) writes the answer using only those "
        "chunks, in the language of your question.\n\n"
        "Every answer cites the document and page it came from, and the assistant states clearly "
        "when the answer is not present in the indexed documents."
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
