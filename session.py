"""
session.py
----------
Backend orchestration layer: sits between the RAG chain (chains.py) and
any front end (app.py, or later a FastAPI/Flask API). Owns chat-history
management, per-user session isolation, and turn-level bookkeeping
(sources returned per turn, history length control).

Why this file exists instead of calling get_rag_chain() directly from
app.py:
- Building the vector store + RAG chain is expensive (loads the embedding
  model, connects to Groq) and must happen ONCE, not on every user turn.
  This file builds it once and hands out cheap per-session wrappers.
- Multi-turn chat needs chat_history threaded through every call in
  LangChain's message format (HumanMessage/AIMessage pairs) — that
  bookkeeping belongs here, not duplicated in the UI code.
- Unbounded chat_history is a real problem we already hit once: Groq's
  free-tier token-per-minute (TPM) budget (8000 TPM) can be exceeded by a
  long conversation's accumulated history alone, even before the current
  question is added. This file caps history length so old turns age out
  automatically instead of silently blowing the TPM budget deeper into a
  conversation.
- Keeping this separate from app.py means a future non-Streamlit front
  end (a REST API, a CLI, WhatsApp bot, whatever) can reuse the exact
  same session logic without touching Streamlit at all.

In-memory only for now (a plain dict keyed by session_id). Per the
project's workplan this is the Week 7 "backend orchestration, session
handling" deliverable; swapping the in-memory store for Redis-backed
history (as LawGlance's lawglance_main.py does) is a later, backward
compatible change — only _SESSIONS' storage would need to change, not
this file's public interface.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Max number of past (human, ai) turn-pairs kept in chat_history. Older
# turns are dropped first. This bounds both the tokens sent to Groq per
# call (protects the TPM budget) and memory usage for long-running
# sessions. 6 turns is generous for a multi-turn follow-up flow
# (pronoun/topic-switch questions in test_cases.py only ever look back
# 1 turn) while keeping worst-case history small.
MAX_HISTORY_TURNS = 6


@dataclass
class Turn:
    """One question/answer exchange, kept for display and citation purposes."""
    question: str
    answer: str
    sources: List[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    """
    One user's ongoing conversation. Holds LangChain-format chat_history
    (for feeding back into the chain) plus a parallel list of Turn objects
    (for the UI to render, including source citations the raw
    chat_history doesn't carry).
    """
    session_id: str
    rag_chain: object  # the chain returned by chains.get_rag_chain()
    chat_history: List[BaseMessage] = field(default_factory=list)
    turns: List[Turn] = field(default_factory=list)

    def ask(self, question: str) -> Turn:
        """
        Run one turn: contextualize the question against history, retrieve,
        generate an answer, then update history and return the Turn.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        result = self.rag_chain.invoke(
            {"input": question, "chat_history": self.chat_history}
        )
        answer = result["answer"]
        retrieved_docs = result.get("context", [])

        sources = [
            {
                "act": doc.metadata.get("act", "Unknown Act"),
                "section": doc.metadata.get("section", "Unknown Section"),
                "excerpt": doc.page_content[:300],
            }
            for doc in retrieved_docs
        ]

        turn = Turn(question=question, answer=answer, sources=sources)
        self.turns.append(turn)

        # Extend LangChain-format history, then trim to the cap.
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.chat_history) > max_messages:
            dropped = len(self.chat_history) - max_messages
            self.chat_history = self.chat_history[dropped:]
            logger.info(
                "Session %s: trimmed %d oldest message(s) to stay within "
                "MAX_HISTORY_TURNS=%d.",
                self.session_id, dropped, MAX_HISTORY_TURNS,
            )

        return turn

    def reset(self) -> None:
        """Clear history and turns, keeping the same session_id and chain."""
        self.chat_history = []
        self.turns = []


class SessionManager:
    """
    Owns the (expensive, shared, stateless) RAG chain and hands out cheap
    per-session ChatSession wrappers around it. One SessionManager should
    be created once per process and reused — in Streamlit that means
    caching it with @st.cache_resource (see app.py).
    """

    def __init__(self, vector_store):
        from chains import get_rag_chain  # local import avoids a hard

        # dependency on chains.py for callers that only need the dataclasses
        self._rag_chain = get_rag_chain(vector_store)
        self._sessions: dict[str, ChatSession] = {}
        logger.info("SessionManager ready (RAG chain built once, shared across sessions).")

    def create_session(self, session_id: Optional[str] = None) -> ChatSession:
        session_id = session_id or str(uuid.uuid4())
        session = ChatSession(session_id=session_id, rag_chain=self._rag_chain)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> ChatSession:
        return self._sessions.get(session_id) or self.create_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


if __name__ == "__main__":
    from vector_store import get_vector_store

    vs = get_vector_store()
    manager = SessionManager(vs)
    session = manager.create_session()

    turn1 = session.ask("What is the punishment for theft?")
    print("Q1:", turn1.question)
    print("A1:", turn1.answer)
    print("Sources:", [s["section"] for s in turn1.sources])

    turn2 = session.ask("What about if it happens at night?")
    print("\nQ2:", turn2.question)
    print("A2:", turn2.answer)