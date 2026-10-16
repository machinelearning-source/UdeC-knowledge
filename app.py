import streamlit as st
from core.pipeline import query_rag_system

st.set_page_config(page_title="UdeC Knowledge Hub", page_icon="🎓", layout="centered")

st.title("🎓 UdeC-Knowledge")
st.subheader("Intelligent Knowledge Management System")
st.markdown("**Pilot Project:** Smart search of institutional regulations and university guidelines.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your RAG assistant. What would you like to ask today about the university regulations or guidelines?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Document sources"):
                for source in message["sources"]:
                    st.markdown(source)

if prompt := st.chat_input("Type your question here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    assistant_response, sources = query_rag_system(prompt)

    response_content = assistant_response
    shown_sources = [] if assistant_response.startswith("Error:") else sources

    with st.chat_message("assistant"):
        st.markdown(response_content)
        if shown_sources:
            with st.expander("Document sources"):
                for source in shown_sources:
                    st.markdown(source)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_content, "sources": shown_sources}
    )
