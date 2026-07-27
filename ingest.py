"""
ingest.py

Builds the persistent vector store from raw statute text.
Run this once after adding/updating files in data/, before starting app.py.

Pipeline: collect -> clean -> chunk -> embed -> index (ChromaDB)

Usage:
    python ingest.py
"""

import os
import glob
import logging

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Config ----
DATA_DIR = "data"
PERSIST_DIR = "chroma_db_legal"
CHUNK_SIZE = 1000        # characters per chunk; tune based on section length
CHUNK_OVERLAP = 150      # overlap so section context isn't cut mid-sentence


def load_raw_documents(data_dir: str) -> list[Document]:
    """
    Reads every .txt file in data_dir into a Document, tagging each with
    its source filename as metadata (used later for citing the act/section).
    """
    documents = []
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not txt_files:
        logger.warning(
            "No .txt files found in '%s'. Add cleaned statute text there first "
            "(see data/README.md for sourcing instructions).", data_dir
        )
        return documents

    for path in txt_files:
        act_name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(Document(page_content=text, metadata={"source": act_name}))
        logger.info("Loaded %s (%d chars)", act_name, len(text))

    return documents


def clean_text(text: str) -> str:
    """
    Basic normalization: strip repeated whitespace, page-number artifacts,
    and stray control characters left over from PDF-to-text extraction.

    Extend this as you find real artifacts in your specific source PDFs —
    every government PDF has slightly different extraction quirks.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    return "\n".join(lines)


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits each act into overlapping chunks. Starting point uses generic
    recursive character splitting; a stronger v2 should split on
    'Section <number>' boundaries so each chunk maps to one legal section.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\nSection ", "\n\n", "\n", ". ", " "],
    )

    chunks = []
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        split_chunks = splitter.split_documents([doc])
        chunks.extend(split_chunks)

    logger.info("Produced %d chunks from %d source documents", len(chunks), len(documents))
    return chunks


def build_vector_store(chunks: list[Document], persist_dir: str) -> None:
    """
    Embeds all chunks and writes them into a persistent Chroma vector store.
    """
    if not chunks:
        logger.error("No chunks to index — check that data/ contains source files.")
        return

    embeddings = OpenAIEmbeddings()

    logger.info("Embedding %d chunks and writing to '%s'...", len(chunks), persist_dir)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    logger.info("Vector store build complete.")


def main():
    documents = load_raw_documents(DATA_DIR)
    chunks = chunk_documents(documents)
    build_vector_store(chunks, PERSIST_DIR)


if __name__ == "__main__":
    main()
