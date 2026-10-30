import base64
import csv
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core.calendar_names import CALENDAR_OFFICIAL_NAMES
from core.pipeline import (
    cited_sources,
    knowledge_stats,
    normalize_citations,
    prepare_query,
    stream_answer,
    warm_up,
)

ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "knowbot-logo.png"
AVATAR_PATH = ASSETS_DIR / "knowbot-avatar.png"
ASSISTANT_AVATAR = str(AVATAR_PATH) if AVATAR_PATH.exists() else "🤖"

st.set_page_config(
    page_title="KnowBot",
    page_icon=ASSISTANT_AVATAR,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Copy the secrets (GROQ_API_KEY on Streamlit Cloud) into the environment. Only when a
# secrets.toml exists: reading st.secrets without one shows a "No secrets found" warning.
if st.secrets.load_if_toml_exists():
    for _secret_key, _secret_value in st.secrets.items():
        os.environ.setdefault(_secret_key, str(_secret_value))


@st.cache_resource(show_spinner="Loading the search engine (only once)...")
def _warm_engine():
    warm_up()
    return True


@st.cache_data(show_spinner=False)
def _knowledge_stats():
    return knowledge_stats()


@st.cache_data
def _image_as_data_uri(path):
    if not Path(path).exists():
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


NAVY_DARK = "#071a33"
NAVY = "#0b2a4f"
BLUE = "#0e4a80"
CYAN = "#22d3ee"
CYAN_DARK = "#0891b2"
INK = "#1e293b"
PAGE_BG = "#eef4f9"
SURFACE = "#f5f9fc"
BORDER = "rgba(11, 42, 79, 0.14)"
SOFT_TEXT = "#a5d8f0"
BRAND_GRADIENT = f"linear-gradient(100deg, {NAVY_DARK} 0%, {NAVY} 55%, {BLUE} 100%)"

CALENDAR_YEAR = "2026"
SOURCE_PATTERN = re.compile(r"^(?:\[(?P<number>\d+)\]\s*)?PDF:\s*(?P<name>.+?)\s*\(Page (?P<page>\d+)\)$")
MAX_SOURCE_TAGS = 6
FEEDBACK_PATH = Path(__file__).parent / "logs" / "feedback.csv"
FEEDBACK_FIELDS = ["timestamp", "rating", "question", "answer", "sources"]

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "🤖 **Welcome to KnowBot!**\n\n"
        "Ask. Learn. Know.\n\n"
        "I can help you with:\n\n"
        "- Academic regulations\n"
        "- Student rights and duties\n"
        "- Homologation procedures\n"
        "- Academic calendars\n"
        "- Institutional documentation\n\n"
        "How can I help you today?"
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
CALENDAR_SEARCH_QUERY = "2026 academic calendar {official} {option} dates"

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    [data-testid="stToolbar"] {{visibility: hidden;}}
    [data-testid="stDecoration"] {{display: none;}}
    footer {{visibility: hidden;}}
    [data-testid="stHeader"] {{background: transparent;}}

    .stApp {{background: {PAGE_BG} !important;}}
    [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {{background: transparent !important;}}
    .block-container {{padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1080px;}}
    [data-testid="stBottomBlockContainer"] {{max-width: 1080px; margin: 0 auto;}}

    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {{
        background: {SURFACE} !important;
    }}
    [data-testid="stSidebar"] {{border-right: 4px solid {CYAN};}}
    [data-testid="stSidebar"] * {{color: {NAVY};}}
    [data-testid="stSidebar"] h4 {{color: {NAVY}; font-weight: 800;}}
    .kb-brand {{
        background: {BRAND_GRADIENT};
        border-left: 5px solid {CYAN};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 18px;
    }}
    [data-testid="stSidebar"] .kb-brand h2 {{margin: 0; font-size: 1.45rem; color: #ffffff;}}
    [data-testid="stSidebar"] .kb-brand h2 span {{color: {CYAN};}}
    [data-testid="stSidebar"] .kb-brand p {{
        margin: 4px 0 0; font-size: 0.78rem; letter-spacing: 3px; color: {SOFT_TEXT};
    }}

    .kb-hero {{
        background: {BRAND_GRADIENT};
        border-bottom: 7px solid {CYAN};
        border-radius: 20px;
        padding: 26px 34px;
        color: #ffffff;
        box-shadow: 0 12px 28px rgba(7, 26, 51, 0.25);
        margin-bottom: 6px;
    }}
    .kb-hero img {{height: 120px; display: block;}}
    .kb-hero h1 {{color: #ffffff; font-size: 2.6rem; margin: 0;}}
    .kb-hero h1 span {{color: {CYAN};}}
    .kb-hero p {{margin: 4px 0 0; color: {SOFT_TEXT}; letter-spacing: 4px; font-weight: 600;}}

    .kb-tagline {{color: {NAVY}; font-weight: 600; font-size: 0.95rem; margin: 10px 0 18px;}}

    .kb-card-head {{
        background: {BRAND_GRADIENT};
        border-left: 5px solid {CYAN};
        border-radius: 12px;
        padding: 10px 14px;
        color: #ffffff;
        margin-bottom: 6px;
    }}
    .kb-card-head .title {{font-weight: 700; font-size: 0.98rem; display: block;}}
    .kb-card-head .count {{font-size: 0.85rem; color: {SOFT_TEXT};}}

    [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"]:has(.kb-card-head):not(:has([data-testid="stVerticalBlockBorderWrapper"] .kb-card-head)) {{
        border-radius: 16px;
        border: 1px solid {BORDER};
        background: #ffffff;
        box-shadow: 0 6px 18px rgba(7, 26, 51, 0.08);
    }}

    .stButton > button,
    [data-testid="stSidebar"] .stButton > button {{
        background: {NAVY} !important;
        border: 2px solid {NAVY} !important;
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(7, 26, 51, 0.18);
        transition: all 0.15s ease;
    }}
    .stButton > button,
    .stButton > button *,
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button * {{
        color: #ffffff !important;
    }}
    .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {BLUE} !important;
        border-color: {CYAN} !important;
    }}

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        color: {NAVY};
    }}
    /* Hand cursor on the dropdowns and their options (the select is searchable, so it showed a text cursor). */
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    [data-baseweb="popover"] li[role="option"],
    [data-baseweb="popover"] li[role="option"] *,
    [data-testid="stPopover"] button {{
        cursor: pointer !important;
    }}

    [data-testid="stChatMessage"] {{
        background: #ffffff;
        border: 1px solid {BORDER};
        border-left: 6px solid {CYAN};
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(7, 26, 51, 0.06);
        padding: 4px 10px;
        margin-bottom: 8px;
        color: {INK};
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        border-left-color: {BLUE};
        background: {SURFACE};
    }}
    [data-testid="chatAvatarIcon-user"],
    [data-testid="stChatMessageAvatarUser"] {{background: {BLUE} !important; color: #ffffff !important;}}
    [data-testid="stChatMessageAvatarUser"] * {{color: #ffffff !important;}}
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {{
        border: 2px solid {BLUE};
        border-radius: 12px;
        background: #ffffff;
    }}
    [data-testid="stExpander"] summary {{color: {NAVY}; font-weight: 600;}}
    [data-testid="stExpander"] details {{border-color: {BORDER};}}
    .kb-hero {{display: flex; align-items: center; justify-content: space-between; gap: 18px; flex-wrap: wrap;}}
    .kb-stats {{display: flex; gap: 12px; flex-wrap: wrap;}}
    .kb-stat {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(34, 211, 238, 0.35);
        border-radius: 14px;
        padding: 8px 16px;
        text-align: center;
        min-width: 104px;
    }}
    .kb-stat b {{display: block; font-size: 1.6rem; line-height: 1.15; color: {CYAN};}}
    .kb-stat span {{font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; color: {SOFT_TEXT};}}

    [data-testid="stChatMessage"] [data-testid="stPopover"] button {{
        background: #ffffff !important;
        border: 1.5px solid {BLUE} !important;
        border-radius: 999px !important;
        color: {NAVY} !important;
        font-weight: 600;
        padding: 2px 14px;
        min-height: 2.1rem;
        white-space: nowrap;
    }}
    [data-testid="stChatMessage"] [data-testid="stPopover"] button * {{color: {NAVY} !important;}}
    [data-testid="stChatMessage"] [data-testid="stPopover"] button:hover {{
        background: {NAVY} !important;
        border-color: {NAVY} !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stPopover"] button:hover * {{color: #ffffff !important;}}
    [data-testid="stPopoverBody"] {{border-radius: 14px; border: 1px solid {BORDER};}}
    /* Quick-question chips: each as wide as its text, wrapping to a new line instead of overlapping. */
    [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) {{
        flex-wrap: wrap;
        gap: 8px;
    }}
    [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) > [data-testid="stColumn"] {{
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }}

    .kb-sources {{display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 6px 0 4px;}}
    .kb-sources-label {{font-size: 0.78rem; font-weight: 700; color: #5b6b80;}}
    .kb-tag {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        color: {BLUE};
    }}
    .kb-tag::before {{content: "📄 ";}}

    @media (max-width: 640px) {{
        .block-container {{padding-top: 0.8rem;}}
        .kb-hero {{padding: 16px 18px; gap: 12px;}}
        .kb-hero img {{height: 64px;}}
        .kb-stats {{width: 100%; gap: 8px; flex-wrap: nowrap;}}
        .kb-stat {{flex: 1; min-width: 0; padding: 6px 4px;}}
        .kb-stat b {{font-size: 1.15rem;}}
        .kb-stat span {{font-size: 0.6rem; letter-spacing: 1px;}}
    }}
    .kb-footer {{
        margin-top: 26px;
        padding-top: 10px;
        border-top: 3px solid {CYAN};
        color: {NAVY};
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
        <div class="kb-brand">
            <h2>Know<span>Bot</span></h2>
            <p>ASK. LEARN. KNOW.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "show_about" not in st.session_state:
        st.session_state.show_about = False

    st.markdown("#### Quick actions")
    if st.button("Reload page", icon=":material/refresh:", use_container_width=True):
        st.session_state.clear()
        st.query_params["reloaded"] = "1"
        st.rerun()

    if st.button("New conversation", icon=":material/add_comment:", use_container_width=True):
        st.session_state.messages = [dict(WELCOME_MESSAGE)]
        st.session_state.suggestion = None
        st.session_state.flash = "New conversation started."
        st.rerun()

    if st.button("About", icon=":material/info:", use_container_width=True):
        st.session_state.show_about = not st.session_state.show_about

    if st.session_state.show_about:
        st.markdown(
            "**About KnowBot**  \n"
            "This pilot answers questions about the university's official documents with a "
            "Retrieval-Augmented Generation pipeline:\n\n"
            "1. **Ingestion:** the PDFs are read and split into chunks.\n"
            "2. **Embeddings:** FastEmbed (multilingual MiniLM) runs locally.\n"
            "3. **Retrieval:** calendar activities are matched by their official name, and "
            "FAISS + MMR finds the most similar chunks.\n"
            "4. **Generation:** the Groq model `openai/gpt-oss-120b` (or `openai/gpt-oss-20b` when "
            "its daily quota is used up) writes the answer using only those chunks, always in English.\n\n"
            "Every answer cites the document and page it came from and says clearly when the "
            "information is not in the indexed documents."
        )

logo_uri = _image_as_data_uri(str(LOGO_PATH))
if logo_uri:
    hero_brand = f'<img src="{logo_uri}" alt="KnowBot - Ask. Learn. Know.">'
else:
    hero_brand = "<div><h1>🤖 Know<span>Bot</span></h1><p>ASK. LEARN. KNOW.</p></div>"
stats = _knowledge_stats()
hero_stats = "".join(
    f'<div class="kb-stat"><b>{value}</b><span>{label}</span></div>'
    for value, label in [
        (stats["documents"], "Documents"),
        (stats["sections"], "Sections"),
        (CALENDAR_YEAR, "Calendar"),
    ]
)
st.markdown(
    f'<div class="kb-hero">{hero_brand}<div class="kb-stats">{hero_stats}</div></div>',
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


def _avatar_for(role):
    return ASSISTANT_AVATAR if role == "assistant" else None


def _suggest(question, search_query=None, calendar_name=None):
    """Put a quick question in the chat box; it is asked when the user sends it."""
    st.session_state.suggestion = {
        "question": question,
        "search_query": search_query,
        "calendar_name": calendar_name,
    }
    st.session_state.fill_chat_input = True


def _suggest_category():
    chosen = st.session_state.chosen_category
    if chosen:
        _suggest(CATEGORY_QUESTION.format(option=chosen))
        st.session_state.chosen_category = None


def _suggest_course():
    chosen = st.session_state.chosen_course
    if chosen:
        _suggest(COURSE_QUESTION.format(option=chosen))
        st.session_state.chosen_course = None


def _suggest_calendar():
    chosen = st.session_state.chosen_activity
    if chosen:
        official_name = CALENDAR_OFFICIAL_NAMES.get(chosen, chosen)
        _suggest(
            CALENDAR_QUESTION.format(option=chosen),
            CALENDAR_SEARCH_QUERY.format(option=chosen, official=official_name),
            official_name,
        )
        st.session_state.chosen_activity = None


def _fill_chat_input(text):
    """Write text into the chat box (st.chat_input has no value parameter in Streamlit 1.39).

    A zero-height component runs a script in the page that sets the box's value
    the way React expects and focuses it, so the user can edit and send it.
    """
    components.html(
        f"""
        <script>
        const text = {json.dumps(text)};
        const page = window.parent.document;
        function fill(attempts) {{
            const box = page.querySelector('[data-testid="stChatInput"] textarea');
            if (!box) {{
                if (attempts > 0) setTimeout(() => fill(attempts - 1), 100);
                return;
            }}
            const setValue = Object.getOwnPropertyDescriptor(
                window.parent.HTMLTextAreaElement.prototype, "value"
            ).set;
            setValue.call(box, text);
            box.dispatchEvent(new Event("input", {{ bubbles: true }}));
            box.focus();
        }}
        fill(30);
        </script>
        """,
        height=0,
    )


def _render_suggestions():
    """Quick questions under the welcome message: choosing an option writes it in the chat box.

    The question is asked only when the user sends it, so it can be edited first.
    Each selector starts empty and is emptied again, so the first option and a
    repeated option also work.
    """
    category_column, course_column, calendar_column, _ = st.columns([1.15, 1.2, 1.1, 1.3], gap="small")

    with category_column:
        with st.popover(":material/school: Student categories"):
            st.caption(f"{len(STUDENT_CATEGORIES)} student situations · choose one, then send it")
            st.selectbox(
                "Choose a student category",
                STUDENT_CATEGORIES,
                index=None,
                placeholder="Choose a student situation…",
                key="chosen_category",
                on_change=_suggest_category,
                label_visibility="collapsed",
            )

    with course_column:
        with st.popover(":material/bookmark: Course homologation"):
            st.caption(f"{len(HOMOLOGATION_COURSES)} courses · choose one, then send it")
            st.selectbox(
                "Choose a course",
                HOMOLOGATION_COURSES,
                index=None,
                placeholder="Choose a course…",
                key="chosen_course",
                on_change=_suggest_course,
                label_visibility="collapsed",
            )

    with calendar_column:
        with st.popover(":material/calendar_month: Academic calendar"):
            st.caption(f"{len(CALENDAR_ACTIVITIES)} activities of 2026 · choose one, then send it")
            st.selectbox(
                "Choose an academic calendar activity",
                CALENDAR_ACTIVITIES,
                index=None,
                placeholder="Choose an activity…",
                key="chosen_activity",
                on_change=_suggest_calendar,
                label_visibility="collapsed",
            )


def _source_tags(sources):
    """Turn '[2] PDF: name.pdf (Page 3)' entries into unique '[2] name · p. 3' tags."""
    tags = []
    for source in sources:
        match = SOURCE_PATTERN.match(source)
        if match:
            name = os.path.splitext(re.split(r"[\\/]", match.group("name").strip())[-1])[0]
            number = f"[{match.group('number')}] " if match.group("number") else ""
            tag = f"{number}{name} · p. {match.group('page')}"
        else:
            tag = source.strip()
        if tag not in tags:
            tags.append(tag)
    return tags[:MAX_SOURCE_TAGS]


def _render_sources(sources):
    tags = _source_tags(sources)
    if not tags:
        return
    tags_html = "".join(f'<span class="kb-tag">{html.escape(tag)}</span>' for tag in tags)
    st.markdown(
        f'<div class="kb-sources"><span class="kb-sources-label">Sources:</span>{tags_html}</div>',
        unsafe_allow_html=True,
    )


def _save_feedback(index):
    """Append the 👍/👎 rating of one answer to logs/feedback.csv."""
    rating = st.session_state.get(f"feedback_{index}")
    if rating is None:
        return
    message = st.session_state.messages[index]
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    is_new_file = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as feedback_file:
        writer = csv.DictWriter(feedback_file, fieldnames=FEEDBACK_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "rating": "up" if rating == 1 else "down",
                "question": message.get("question", ""),
                "answer": message["content"],
                "sources": " | ".join(message.get("sources", [])),
            }
        )
    st.toast("Thanks for your feedback!")


def _render_feedback(index):
    st.feedback("thumbs", key=f"feedback_{index}", on_change=_save_feedback, args=(index,))


def _write_live(chunks):
    """Show the answer while it is written (st.write_stream needs pyarrow, which is not required here)."""
    placeholder = st.empty()
    placeholder.markdown("_Writing the answer…_ ▌")
    answer = ""
    for chunk in chunks:
        answer += chunk
        placeholder.markdown(normalize_citations(answer) + " ▌")
    answer = normalize_citations(answer)
    placeholder.markdown(answer)
    return answer


def _chat_history():
    """Previous questions and answers, without the welcome message, for follow-up questions."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[1:]
    ]


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"], avatar=_avatar_for(message["role"])):
        st.markdown(message["content"])
        if index == 0 and message["role"] == "assistant":
            _render_suggestions()
        if message.get("sources"):
            _render_sources(message["sources"])
        if index > 0 and message["role"] == "assistant":
            _render_feedback(index)

prompt = st.chat_input("Ask KnowBot your question...")
suggestion = st.session_state.get("suggestion") or {}
if st.session_state.pop("fill_chat_input", False) and suggestion:
    _fill_chat_input(suggestion["question"])

# A sent suggestion keeps its search hints (the official calendar name); an edited one is a typed question.
search_query, calendar_name = prompt, None
if prompt and prompt.strip() == suggestion.get("question"):
    search_query = suggestion.get("search_query") or prompt
    calendar_name = suggestion.get("calendar_name")
if prompt:
    st.session_state.suggestion = None

if prompt:
    history = _chat_history()
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching the documents..."):
            _warm_engine()
            request = prepare_query(prompt, search_query, calendar_name, history)

        if request.error:
            assistant_response = request.error
            st.markdown(assistant_response)
        else:
            assistant_response = _write_live(stream_answer(request))
        shown_sources = cited_sources(assistant_response, request.fragment_sources)
        _render_sources(shown_sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
            "sources": shown_sources,
            "question": request.question,
        }
    )
    st.rerun()

st.markdown(
    '<div class="kb-footer">KnowBot · FastEmbed + FAISS + Groq · UI v3.0</div>',
    unsafe_allow_html=True,
)
