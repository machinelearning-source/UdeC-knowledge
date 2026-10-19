import os

import streamlit as st

from core.pipeline import query_rag_system, warm_up

st.set_page_config(page_title="UdeC Knowledge Hub", page_icon="🎓", layout="centered")

try:
    for _secret_key, _secret_value in st.secrets.items():
        os.environ.setdefault(_secret_key, str(_secret_value))
except Exception:
    pass


@st.cache_resource(show_spinner="Loading the search engine (only once)...")
def _warm_engine():
    warm_up()


_warm_engine()

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
question_columns = st.columns(3)

if question_columns[0].button("Student categories", use_container_width=True):
    st.session_state.pending_question = "What student categories exist at the university?"

if question_columns[1].button("Course homologation", use_container_width=True):
    st.session_state.pending_question = "What is the procedure to homologate a course?"

if question_columns[2].button("Academic calendar", use_container_width=True):
    st.session_state.pending_question = (
        "What does the 2026 academic calendar say about enrollment dates?"
    )

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
