"""
chains.py

Builds the RAG chain that answers legal questions using the vector
store from build_knowledge_base.py.

Uses Groq for chat/generation (ChatGroq) — fast, cheap LLM inference.
Embeddings are handled separately by a LOCAL model (see vector_store.py)
since Groq does not offer an embeddings API. These are two different
jobs done by two different providers:

  - Embeddings (text -> vector, for search):  local sentence-transformers
  - Chat generation (context -> answer):      Groq

Two pieces, chained together:

  1. History-aware retriever
     Takes the latest user question + prior chat history, and asks the
     LLM to rewrite it into a standalone question if it depends on
     earlier context (e.g. "what about a repeat offence?" becomes
     "what is the punishment for a repeat offence of theft?").
     That standalone question is then used to search the vector store.

  2. QA chain ("stuff documents")
     Takes the retrieved chunks, stuffs them into a single context
     block, and asks the LLM to answer the question using ONLY that
     context — this is what keeps answers grounded in the actual
     statute instead of the model's general training knowledge.

Setup:
    Get a free API key from https://console.groq.com
    Add to .env:  GROQ_API_KEY=your-key-here

Usage:
    from vector_store import get_vector_store
    from chains import get_rag_chain

    vector_store = get_vector_store()
    rag_chain = get_rag_chain(vector_store)

    response = rag_chain.invoke({
        "input": "What is the punishment for theft?",
        "chat_history": [],
    })
    print(response["answer"])
"""

import logging

from dotenv import load_dotenv

# NOTE: create_history_aware_retriever / create_retrieval_chain used to live
# in langchain.chains, but LangChain 1.0 moved them into a separate
# 'langchain_classic' package (langchain.chains no longer exists as of 1.0).
# Install both: pip install langchain langchain-classic
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Config ----
# Groq model. "llama-3.1-8b-instant" is fast and cheap — good for iterating.
# "llama-3.3-70b-versatile" is stronger but slower/costlier — swap in for
# final answer quality once the pipeline works end-to-end.
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.2          # low temperature: favor grounded, consistent answers over creativity
RETRIEVAL_K = 4                # how many chunks to retrieve per query
SCORE_THRESHOLD = 0.5          # minimum similarity score to keep a retrieved chunk


# ============================================================
# Prompts
# ============================================================

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question, which might "
    "reference context in the chat history, formulate a standalone "
    "question which can be understood without the chat history. "
    "Do NOT answer the question — just reformulate it if needed, "
    "and otherwise return it as-is."
)

QA_SYSTEM_PROMPT = (
    "You are a legal information assistant for Indian law. Answer the "
    "user's question using ONLY the retrieved context below, which "
    "contains excerpts from Indian statutes. \n\n"
    "Rules:\n"
    "- If the context does not contain enough information to answer, "
    "say so clearly instead of guessing or using outside knowledge.\n"
    "- Where possible, mention which act and section the answer comes "
    "from (this is available in the context).\n"
    "- Explain the answer in plain, accessible language — the user is "
    "not assumed to be a lawyer.\n"
    "- Refer to acts ONLY by the name given in the retrieved context's "
    "metadata. Never substitute a different or older statute name (e.g. do "
    "not say 'Indian Penal Code' when the context is from the Bharatiya "
    "Nyaya Sanhita, 2023, which replaced it).\n"
    "- This is legal information, not legal advice. Do not tell the "
    "user what they personally should do; explain what the law says.\n\n"
    "Retrieved context:\n{context}"
)


# ============================================================
# Chain construction
# ============================================================

def get_llm() -> ChatGroq:
    """Returns the configured Groq chat LLM used for both reformulation and generation."""
    return ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def get_retriever(vector_store, llm):
    """
    Wraps the vector store in a history-aware retriever: incoming
    questions are first rewritten into a standalone form using chat
    history, then used to run similarity search with a score
    threshold so weakly-relevant chunks are filtered out rather than
    force-fed to the LLM.
    """
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVAL_K, "score_threshold": SCORE_THRESHOLD},
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, base_retriever, contextualize_prompt
    )

    return history_aware_retriever


def get_rag_chain(vector_store):
    """
    Builds the full RAG chain: history-aware retrieval -> stuff
    retrieved chunks into context -> LLM generates a grounded answer.

    Returns a chain you can call with:
        chain.invoke({"input": "...", "chat_history": [...]})
    which returns a dict containing at least:
        "answer"  - the generated response
        "context" - the list of retrieved Document chunks used
    """
    llm = get_llm()
    retriever = get_retriever(vector_store, llm)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    logger.info(
        "RAG chain ready (model=%s via Groq, k=%d, score_threshold=%.2f)",
        LLM_MODEL, RETRIEVAL_K, SCORE_THRESHOLD,
    )

    return rag_chain


if __name__ == "__main__":
    # Quick standalone check: python chains.py
    from vector_store import get_vector_store

    vector_store = get_vector_store()
    rag_chain = get_rag_chain(vector_store)

    test_question = "What is the punishment for theft?"
    print(f"\nQuestion: {test_question}\n")

    response = rag_chain.invoke({"input": test_question, "chat_history": []})

    print("Answer:")
    print(response["answer"])

    print("\nRetrieved from:")
    for doc in response["context"]:
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        print(f"  - {source} {section}")
