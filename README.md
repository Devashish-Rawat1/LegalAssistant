# Groq + local embeddings setup

This project uses **two different providers** for two different jobs:

| Job | Provider | Why |
|---|---|---|
| Embeddings (text → vector, for search) | Local — `sentence-transformers` via HuggingFace | Groq has no embeddings API. Runs on your CPU, free, no key. |
| Chat generation (context → answer) | Groq (`ChatGroq`) | Fast, free-tier API, only used for actual answer generation. |

This is normal — most RAG stacks call two different model providers for
these two jobs. `vector_store.py` and `build_knowledge_base.py` never
touch Groq; `chains.py` never touches the embedding model directly.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Get a free Groq API key from https://console.groq.com and add it to `.env`:

```
GROQ_API_KEY=your-key-here
```

No key is needed for embeddings — the first time you run
`build_knowledge_base.py` or `vector_store.py`, it downloads a small
(~80MB) embedding model from HuggingFace and caches it locally. After
that first download, it runs fully offline.

## Running

```bash
python build_knowledge_base.py   # builds chroma_db_legal_bot_part1/ (no Groq needed)
python vector_store.py           # sanity-check the store loads (no Groq needed)
python chains.py                 # asks a test question (needs GROQ_API_KEY)
```

## Important: keep the embedding model consistent

`vector_store.py` and `build_knowledge_base.py` both hardcode:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
```

If you ever change this in one file, change it in the other too. Chroma
stores vectors, not the model that produced them — querying a store
built with one embedding model using a *different* embedding model at
query time silently returns irrelevant results, with no error to warn
you.

## Swapping the Groq model

`chains.py` defaults to `llama-3.1-8b-instant` — fast and good for
iterating while you're building the pipeline. For better answer
quality once everything works end-to-end, swap `LLM_MODEL` in
`chains.py` to `llama-3.3-70b-versatile` (slower, but noticeably
stronger reasoning over legal text).
