"""
vector_store.py

Loads the persistent Chroma vector store from disk.

Uses a LOCAL embedding model (sentence-transformers via HuggingFace),
not OpenAI or Groq — this runs on your own machine, needs no API key,
and costs nothing per call. This matters because Groq does not offer
an embeddings API; embeddings and chat generation are handled by two
completely separate pieces in this project (see chains.py for the
Groq-based chat model).

IMPORTANT: whatever embedding model built chroma_db_legal_bot_part1/
must be the SAME model used here to query it. If build_knowledge_base.py
used 'sentence-transformers/all-MiniLM-L6-v2', this file must too —
otherwise similarity search compares vectors from two different
vector spaces and silently returns irrelevant results.

Usage:
    from vector_store import get_vector_store
    vector_store = get_vector_store()
"""

import os
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = "chroma_db_legal_bot_part1"

# Small, fast, well-regarded general-purpose embedding model.
# Runs locally on CPU, no API key needed. ~80MB download on first use
# (cached afterward). Must match the model used in build_knowledge_base.py.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the local embedding model. Shared by build_knowledge_base.py
    (writing) and this file (reading) so both sides always agree on
    which model produced the vectors.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vector_store(persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    """
    Opens the existing Chroma vector store for querying.

    Raises a clear error if the folder is missing, instead of silently
    creating an empty store (which would make every retrieval return
    nothing, with no obvious reason why).
    """
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"'{persist_directory}' not found in the project root. "
            f"Run build_knowledge_base.py first to create it."
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    count = vector_store._collection.count()
    logger.info("Loaded vector store '%s' with %d indexed chunks", persist_directory, count)

    if count == 0:
        logger.warning(
            "Vector store loaded but contains 0 documents. "
            "Check that '%s' is the correct folder and wasn't copied empty.",
            persist_directory,
        )

    return vector_store


if __name__ == "__main__":
    # Quick standalone check: python vector_store.py
    store = get_vector_store()
    print(f"Vector store ready. Document count: {store._collection.count()}")
