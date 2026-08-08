"""
build_knowledge_base.py

Full data pipeline, in one file, five stages:

  1. Raw statute text     - extract text from bare-act PDFs
  2. Cleaning             - strip page numbers, headers, rejoin broken lines
  3. Chunking             - split on Section/Article boundaries
  4. Embedding            - convert each chunk to a vector (OpenAI)
  5. chroma_db_legal_bot_part1/ - persist vectors + text + metadata to disk

Folder setup expected:
    data/raw_pdfs/
        constitution_of_india.pdf
        bns_2023.pdf
        consumer_protection_act_2019.pdf
        ...

Output:
    chroma_db_legal_bot_part1/   <- the finished, queryable knowledge base

Usage:
    pip install pdfplumber langchain langchain-openai langchain-chroma python-dotenv
    # add OPENAI_API_KEY to a .env file in this folder
    python build_knowledge_base.py
"""

import os
import re
import glob
import logging

import pdfplumber
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Config ----
RAW_PDF_DIR = "data/raw_pdfs"
PERSIST_DIRECTORY = "chroma_db_legal_bot_part1"

MAX_CHUNK_CHARS = 1500
FALLBACK_CHUNK_SIZE = 1000
FALLBACK_CHUNK_OVERLAP = 150

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*THE GAZETTE OF INDIA\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[PART\s+[IVXLC]+.*\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*EXTRAORDINARY\s*$", re.IGNORECASE),
]
SECTION_MARKER_RE = re.compile(
    r"^(Section|Article)\s+(\d+[A-Za-z]?)\.?\s*[-\u2013\u2014]?\s*(.*)$",
    re.MULTILINE,
)


# ============================================================
# Stage 1: Raw statute text (PDF -> raw text)
# ============================================================

def extract_pdf_text(pdf_path: str) -> str:
    """Extracts text page-by-page from a PDF using pdfplumber."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                logger.warning(
                    "Page %d of %s produced no text (likely scanned/image-only).",
                    i + 1, os.path.basename(pdf_path)
                )
    return "\n".join(pages_text)


# ============================================================
# Stage 2: Cleaning (strip noise, rejoin broken lines)
# ============================================================

def is_junk_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if PAGE_NUMBER_RE.match(stripped):
        return True
    return any(p.match(stripped) for p in HEADER_FOOTER_PATTERNS)


def rejoin_broken_lines(lines: list[str]) -> str:
    """Rejoins PDF line-wraps into full sentences, preserving section breaks."""
    section_start_re = re.compile(r"^\s*(Section|SECTION|Article|ARTICLE)\s+\d+")
    joined = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not buffer:
            buffer = stripped
            continue

        starts_new_section = bool(section_start_re.match(stripped))
        buffer_ends_sentence = buffer.rstrip().endswith((".", ";", ":", ")"))

        if starts_new_section or buffer_ends_sentence:
            joined.append(buffer)
            buffer = stripped
        else:
            buffer = buffer + " " + stripped

    if buffer:
        joined.append(buffer)

    return "\n".join(joined)


def clean_text(raw_text: str) -> str:
    lines = raw_text.splitlines()
    lines = [line for line in lines if not is_junk_line(line)]
    text = rejoin_broken_lines(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# Stage 3: Chunking (split on Section/Article boundaries)
# ============================================================

def fallback_split(text: str) -> list[str]:
    """Generic paragraph-based splitter for text with no section markers."""
    if len(text) <= FALLBACK_CHUNK_SIZE:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) <= FALLBACK_CHUNK_SIZE:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)

    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev_tail = chunks[i - 1][-FALLBACK_CHUNK_OVERLAP:]
            overlapped.append(prev_tail + " " + chunk)
    return overlapped


def split_on_sections(text: str, act_name: str) -> list[dict]:
    """Splits text at each Section/Article marker into whole-section chunks."""
    matches = list(SECTION_MARKER_RE.finditer(text))
    if not matches:
        return []

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue

        section_label = f"{match.group(1)} {match.group(2)}"

        if len(section_text) > MAX_CHUNK_CHARS:
            for j, sub in enumerate(fallback_split(section_text)):
                chunks.append({
                    "text": sub,
                    "metadata": {"source": act_name, "section": section_label, "part": j + 1},
                })
        else:
            chunks.append({
                "text": section_text,
                "metadata": {"source": act_name, "section": section_label},
            })
    return chunks


def chunk_act(text: str, act_name: str) -> list[dict]:
    chunks = split_on_sections(text, act_name)
    if chunks:
        logger.info("%s: split into %d section-aligned chunks", act_name, len(chunks))
        return chunks

    logger.warning(
        "%s: no 'Section N' / 'Article N' markers found — using fallback splitter.",
        act_name,
    )
    return [
        {"text": c, "metadata": {"source": act_name}}
        for c in fallback_split(text)
    ]


# ============================================================
# Stages 4 + 5: Embedding + storing in Chroma
# ============================================================

def embed_and_store(documents: list[Document], persist_directory: str) -> None:
    if not documents:
        logger.error("No documents to embed — nothing to store.")
        return

    embeddings = OpenAIEmbeddings()

    logger.info(
        "Embedding %d chunks and writing to '%s' — this calls the OpenAI "
        "embeddings API and will incur API cost proportional to chunk count.",
        len(documents), persist_directory,
    )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    logger.info(
        "Done. '%s' now contains the full embedded, searchable knowledge base.",
        persist_directory,
    )


# ============================================================
# Pipeline runner: PDF -> raw -> cleaned -> chunked -> embedded -> stored
# ============================================================

def main():
    pdf_files = glob.glob(os.path.join(RAW_PDF_DIR, "*.pdf"))

    if not pdf_files:
        logger.warning(
            "No PDFs found in '%s'. Download bare-act PDFs from India Code "
            "(https://www.indiacode.nic.in) and place them there first.",
            RAW_PDF_DIR,
        )
        return

    all_documents: list[Document] = []

    for pdf_path in pdf_files:
        act_name = os.path.splitext(os.path.basename(pdf_path))[0]
        logger.info("Processing: %s", act_name)

        # Stage 1
        raw_text = extract_pdf_text(pdf_path)
        logger.info("  Stage 1 (extract): %d chars", len(raw_text))

        # Stage 2
        cleaned = clean_text(raw_text)
        logger.info("  Stage 2 (clean):   %d chars -> %d chars", len(raw_text), len(cleaned))

        # Stage 3
        chunks = chunk_act(cleaned, act_name)
        logger.info("  Stage 3 (chunk):   %d chunks", len(chunks))

        for chunk in chunks:
            metadata = {k: v for k, v in chunk["metadata"].items() if v is not None}
            all_documents.append(Document(page_content=chunk["text"], metadata=metadata))

    logger.info("Total chunks across all acts: %d", len(all_documents))

    # Stages 4 + 5
    embed_and_store(all_documents, PERSIST_DIRECTORY)


if __name__ == "__main__":
    main()