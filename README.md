# ⚖️ LegalAssistant — AI-Powered RAG-Based Legal Assistant

A free, AI-powered legal assistant that uses Retrieval-Augmented Generation (RAG) to answer legal questions in plain language, grounded in the actual text of Indian statutes.

This project is a B.Tech Major Project (CS218), Dept. of CS&E, Jaypee University of Engineering & Technology, Guna — built as an original implementation studied against and inspired by the open-source [LawGlance](https://github.com/lawglance/lawglance) project.

**Team:** Devashish Rawat · Akshit Shukla · Ayshmaan Kumar

---

## 📚 Legal Coverage (Phase 1)

- The Indian Constitution
- The Bharatiya Nyaya Sanhita, 2023
- The Bharatiya Nagarik Suraksha Sanhita, 2023
- The Bharatiya Sakshya Adhiniyam, 2023
- The Consumer Protection Act, 2019
- The Motor Vehicles Act, 1988
- The Information Technology Act, 2000

*(Coverage list to be finalized after Week 1 data collection — see `data/README.md`.)*

---

## 🧠 How It Works

1. **Data collection** — raw statute text is sourced from official government sources (India Code) and stored in `data/`.
2. **Cleaning & chunking** — text is cleaned and split into section-level chunks (`ingest.py`).
3. **Embedding & indexing** — chunks are embedded and stored in a persistent ChromaDB vector store.
4. **Query contextualization** — user questions are reformulated into standalone queries using chat history.
5. **Retrieval** — the reformulated query is matched against the vector store (similarity search, top-k + score threshold).
6. **LLM generation** — retrieved chunks + system prompt + chat history are passed to the LLM to generate a grounded answer.
7. **Caching & session** — answers and chat history are cached/persisted via Redis.
8. **UI** — the answer is rendered in a chat interface.

---

## 💻 Developer Quick Start

1. **Clone the repository**
   ```
   git clone https://github.com/<your-org-or-username>/LegalAssistant.git
   cd LegalAssistant
   ```

2. **Create a virtual environment & install dependencies**
   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key**

   Copy `.env.example` to `.env` and add your key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Build the vector store** (run once, or whenever `data/` changes)
   ```
   python ingest.py
   ```

5. **Run the application**
   ```
   streamlit run app.py
   ```

6. **Access the app** at `http://127.0.0.1:8501`

---

## 🗄️ Optional: Redis Caching

Recommended for faster repeated-query performance and persistent chat history.

**Ubuntu/Linux:** `sudo apt-get install redis-server`
**macOS:** `brew install redis`
**Windows:** use WSL, then follow the Linux steps.

Start it with `redis-server`, verify with `redis-cli ping` (expect `PONG`). The app connects to `redis://localhost:6379/0` by default.

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| LLM | OpenAI (pluggable) |
| Embeddings | OpenAI Embeddings |
| Vector store | ChromaDB |
| Caching / sessions | Redis |
| UI | Streamlit (initial), React/Next.js (planned) |

---

## 📁 Project Structure

```
LegalAssistant/
├── data/                  # raw + cleaned statute text (see data/README.md)
├── chroma_db_legal/       # generated vector store (git-ignored)
├── config/                # constants: model name, top-k, score threshold
├── ingest.py              # data cleaning, chunking, embedding, indexing
├── chains.py               # RAG chain construction (retriever + QA chain)
├── legal_assistant_main.py # session handling, caching, conversational wrapper
├── cache.py                 # Redis wrapper
├── prompts.py                # SYSTEM_PROMPT, QA_PROMPT
├── app.py                     # Streamlit chat UI
├── docs/                       # architecture diagrams, screenshots
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚠️ Disclaimer

This tool is an educational/major-project prototype and provides legal information, not legal advice. It does not replace consultation with a qualified lawyer.

---

## 📄 Reference

Architecture and approach studied from [lawglance/lawglance](https://github.com/lawglance/lawglance) (Apache-2.0), re-implemented independently for this project.
