"""
retrieval.py

Hybrid search + cross-encoder reranking on top of the Chroma vector store.

Pipeline (mode="hybrid_rerank", the default):

    query
      ├─ vector search  (Chroma, meaning-based)      -> top POOL_SIZE chunks
      └─ BM25 search    (keyword, exact-term based)  -> top POOL_SIZE chunks
                │
                ▼
    Reciprocal Rank Fusion  -> one merged, de-duplicated candidate list
                │
                ▼
    cross-encoder reranker  -> re-scores each (query, chunk) pair jointly
                │
                ▼
    top k chunks  -> LLM

Why both: embeddings catch paraphrases ("stole my bicycle" ~ theft) but blur exact
tokens ("66C", "dishonestly", "section 185"); BM25 is the opposite. The reranker then
fixes ordering, so the few chunks that reach the LLM are the best ones.

Modes (see MODES): "vector" is the old behaviour, "hybrid" skips the reranker.
Override the default with the RETRIEVAL_MODE environment variable.

Usage:
    from retrieval import HybridSearcher
    searcher = HybridSearcher(vector_store)
    docs = searcher.search("punishment for theft", k=4)
"""

import logging
import math
import os
import re
from collections import Counter, defaultdict

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MODES = ("vector", "hybrid", "hybrid_rerank")
DEFAULT_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_rerank")

POOL_SIZE = 20          # candidates taken from EACH retriever before fusion
RRF_K = 60              # standard RRF damping constant; larger = flatter rank weighting
RERANK_POOL = 20        # fused candidates the cross-encoder re-scores
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Heavier/better alternative: "BAAI/bge-reranker-base"

BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i if in is it of on or that the "
    "this to was what when where which who whom why will with my me do does can".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _doc_key(doc: Document) -> tuple:
    """Identity of a chunk, used to merge the two result lists and drop duplicates."""
    m = doc.metadata
    return (m.get("source"), m.get("section"), m.get("part"), doc.page_content[:80])


class BM25Index:
    """Minimal Okapi BM25 over a fixed list of Documents (no extra dependency)."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            # source + section are indexed too, so "section 185" / "motor vehicles" match
            text = f"{d.metadata.get('source', '')} {d.metadata.get('section', '')} {d.page_content}"
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.n = len(docs)
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 0.0

    def search(self, query: str, k: int) -> list[Document]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.doc_len[i] / self.avg_len)
                scores[i] += idf * tf * (BM25_K1 + 1) / norm
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self.docs[i] for i in top]


def reciprocal_rank_fusion(rankings: list[list[Document]]) -> list[Document]:
    """Merge ranked lists: score(doc) = sum over lists of 1 / (RRF_K + rank)."""
    scores: dict[tuple, float] = defaultdict(float)
    first_seen: dict[tuple, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            key = _doc_key(doc)
            scores[key] += 1.0 / (RRF_K + rank)
            first_seen.setdefault(key, doc)
    return [first_seen[k] for k in sorted(scores, key=scores.get, reverse=True)]


class HybridSearcher:
    def __init__(self, vector_store, mode: str = DEFAULT_MODE):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.vector_store = vector_store
        self.mode = mode
        self._bm25: BM25Index | None = None
        self._reranker = None
        self._reranker_failed = False

    # ---- lazy components ----
    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            data = self.vector_store.get(include=["documents", "metadatas"])
            docs = [Document(page_content=text, metadata=meta or {})
                    for text, meta in zip(data["documents"], data["metadatas"])]
            self._bm25 = BM25Index(docs)
            logger.info("BM25 index built over %d chunks", len(docs))
        return self._bm25

    @property
    def reranker(self):
        if self._reranker is None and not self._reranker_failed:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(RERANK_MODEL)
                logger.info("Loaded reranker %s", RERANK_MODEL)
            except Exception as e:  # noqa: BLE001 - degrade to hybrid instead of crashing
                self._reranker_failed = True
                logger.warning("Reranker unavailable (%s: %s); falling back to hybrid order",
                               type(e).__name__, e)
        return self._reranker

    # ---- search ----
    def search(self, query: str, k: int = 4) -> list[Document]:
        if self.mode == "vector":
            return self.vector_store.similarity_search(query, k=k)

        pool = max(POOL_SIZE, k)
        vector_hits = self.vector_store.similarity_search(query, k=pool)
        keyword_hits = self.bm25.search(query, pool)
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits])

        if self.mode == "hybrid":
            return fused[:k]

        candidates = fused[:max(RERANK_POOL, k)]
        reranker = self.reranker
        if reranker is None:
            return candidates[:k]
        scores = reranker.predict([(query, d.page_content) for d in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:k]]
