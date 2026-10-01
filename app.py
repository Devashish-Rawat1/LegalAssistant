"""
app.py
------
Streamlit chat interface for LegalAssistant.

Run with:
    streamlit run app.py

Requires:
    - GROQ_API_KEY available either in a .env file in the repo root
      (GROQ_API_KEY=gsk_...) or as a real environment variable. The
      sidebar text box is only a manual fallback for a quick one-off test
      — anything typed there is NOT saved to .env and has to be re-typed
      next run.
    - chroma_db_legal_bot_part1/ already built (run
      build_knowledge_base.py first if it doesn't exist yet).
"""

import os

import streamlit as st
from dotenv import load_dotenv

from session import ChatSession, SessionManager
from vector_store import get_vector_store

# Load GROQ_API_KEY (and anything else) from a .env file in the repo root,
# if one exists, into the process environment. This runs once per server
# process (Streamlit re-executes this script on every rerun, but that's
# fine — load_dotenv() is cheap and idempotent). If GROQ_API_KEY is
# already set as a real environment variable, override=False (the
# default) leaves it alone rather than clobbering it with .env's value.
load_dotenv()

st.set_page_config(page_title="LegalAssistant", page_icon="⚖️", layout="centered")


# ---------------------------------------------------------------------------
# Expensive, shared resources: built once per server process (not per user,
# not per message) and cached by Streamlit across reruns/sessions.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading legal knowledge base…")
def load_session_manager() -> SessionManager:
    vector_store = get_vector_store()
    return SessionManager(vector_store)


def get_chat_session() -> ChatSession:
    """
    Return this browser tab's ChatSession, creating it on first load.
    st.session_state is already per-user (Streamlit gives every browser
    tab its own session_state dict), so we don't need SessionManager's
    session_id lookup here — we just create one session per tab and hold
    it directly.
    """
    if "chat_session" not in st.session_state:
        manager = load_session_manager()
        st.session_state.chat_session = manager.create_session()
    return st.session_state.chat_session


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚖️ LegalAssistant")
    st.caption("An AI assistant for Indian statutory law, built on BNS, BNSS, "
               "BSA, the Constitution, and other acts.")

    if os.environ.get("GROQ_API_KEY"):
        st.caption("✅ Groq API key loaded from .env")
    else:
        st.caption("⚠️ No .env file found (or it has no GROQ_API_KEY) — "
                   "enter a key manually below for this session only.")
        key_input = st.text_input("Groq API key", type="password")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input

    if st.button("🔄 New conversation"):
        if "chat_session" in st.session_state:
            st.session_state.chat_session.reset()
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ This tool provides information based on the indexed statutes. "
        "It is not a substitute for advice from a qualified lawyer."
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("LegalAssistant")

if not os.environ.get("GROQ_API_KEY"):
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

try:
    chat_session = get_chat_session()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

# Replay history on every rerun (Streamlit reruns the whole script on each
# interaction; chat_session.turns is what survives across reruns).
for turn in chat_session.turns:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        st.markdown(turn.answer)
        if turn.sources:
            with st.expander(f"📖 Sources ({len(turn.sources)})"):
                for src in turn.sources:
                    st.markdown(f"**{src['act']} — {src['section']}**")
                    st.caption(src["excerpt"] + "…")

# New question.
question = st.chat_input("Ask a question about Indian law…")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                turn = chat_session.ask(question)
            except Exception as e:
                st.error(f"Something went wrong: {type(e).__name__}: {e}")
                st.stop()
        st.markdown(turn.answer)
        if turn.sources:
            with st.expander(f"📖 Sources ({len(turn.sources)})"):
                for src in turn.sources:
                    st.markdown(f"**{src['act']} — {src['section']}**")
                    st.caption(src["excerpt"] + "…")