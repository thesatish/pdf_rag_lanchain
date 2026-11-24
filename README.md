# Intelligent PDF RAG System with Gemini & LangChain

An enterprise-grade, multi-phase Document Question-Answering (RAG) system powered by **LangChain**, **Google Gemini** (`gemini-3.8-flash` & `gemini-embedding-2`), **ChromaDB**, **FastAPI**, **Angular 19**, and **Streamlit**.

Both interfaces are available and can be run simultaneously or independently:
1. **Modern Angular SPA Frontend (`frontend/`) + FastAPI Backend (`pdf-rag-gemini/server.py`)**
2. **Classic Streamlit Web App (`pdf-rag-gemini/app.py`)**

---

## 5-Phase Learning Roadmap Implemented

1. **Phase 1 — Refactor Current RAG:** Decoupled Ingestion, Retrieval, and Generation modules. Reuses existing Chroma database collections on startup without re-ingesting PDFs.
2. **Phase 2 — Full Document Management:** Multi-PDF ingestion, batch uploads, updates/replacements, deletion, metadata enrichment, and SHA-256 duplicate prevention.
3. **Phase 3 — Advanced Retrieval & Telemetry:** Maximal Marginal Relevance (MMR with tunable $\lambda_{mult}$ and $fetch\_k$), similarity score thresholds, document metadata filtering, and diagnostic telemetry.
4. **Phase 4 — RAG Evaluation:** Automated synthetic benchmark generation (triplets of Question, Ground-Truth Answer, Source Chunk), measuring Recall@k, Context Relevance, Faithfulness (no hallucination), Answer Relevance, and Token F1.
5. **Phase 5 — Advanced RAG Techniques:** Multi-query rewriting, hybrid search (Sparse BM25 + Dense Chroma with Reciprocal Rank Fusion), LLM-based reranking, and contextual compression.

---

## Project Structure

```text
.
├── pdf-rag-gemini/               # Python Backend & Streamlit Application
│   ├── rag_modules/              # Modular Core RAG Architecture
│   │   ├── __init__.py           # Unified exports
│   │   ├── config.py             # Settings, models, singleton Chroma client
│   │   ├── ingestion.py          # Phase 1 & 2: Loaders, chunkers, SHA-256 hashes, CRUD
│   │   ├── retrieval.py          # Phase 1 & 3: Similarity, MMR, threshold cutoffs, telemetry
│   │   ├── generation.py         # Phase 1: LCEL generation, citations, audio transcription
│   │   ├── advanced_rag.py       # Phase 5: Query rewriting, BM25 hybrid search, RRF, reranking, compression
│   │   └── evaluation.py         # Phase 4: Synthetic QA generation, metrics (Recall, Faithfulness, F1)
│   ├── server.py                 # FastAPI REST API Backend (Port 8000)
│   ├── app.py                    # Streamlit Web App (Port 8501)
│   ├── rag_pipeline.py           # Facade orchestrator for backward compatibility
│   ├── test_rag.py               # Core test suite
│   ├── test_extended_phases.py   # Phases 1-5 test suite
│   ├── test_api.py               # FastAPI endpoints test suite
│   ├── requirements.txt          # Python dependencies
│   └── .env                      # API keys and model configurations
│
└── frontend/                     # Modern Angular 19 SPA
    ├── src/
    │   ├── app/
    │   │   ├── models/rag.models.ts         # TypeScript data contracts
    │   │   ├── services/rag-api.service.ts  # HTTP client service consuming FastAPI
    │   │   ├── app.component.ts             # Main dashboard logic
    │   │   ├── app.component.html           # 5-phase responsive UI template
    │   │   └── app.component.css            # Dark/slate glassmorphic theme
    │   └── styles.css
    ├── angular.json
    └── package.json
```

---

## Quick Start Guide

### 1. Environment Setup

```bash
cd pdf-rag-gemini

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python requirements
pip install -r requirements.txt
```

Verify or configure your `.env` file in [`pdf-rag-gemini/.env`](file:///home/l163-tm/Public/python/pdf_rag_lanchain/pdf-rag-gemini/.env):
```env
GOOGLE_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=gemini-embedding-2
LLM_MODEL=gemini-3.8-flash
PERSIST_DIR=./chroma_db
```

---

### Option A: Run FastAPI Backend + Angular Frontend

#### 1. Start FastAPI Backend (Port 8000)
```bash
cd pdf-rag-gemini
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: **`http://localhost:8000/docs`**

#### 2. Start Angular Frontend (Port 4200)
In a separate terminal:
```bash
cd frontend
npm start
```
Access the Angular SPA at: **`http://localhost:4200`**

---

### Option B: Run Streamlit Web Application (Port 8501)

The original Streamlit application remains fully functional:
```bash
cd pdf-rag-gemini
source venv/bin/activate
streamlit run app.py
```
Access the Streamlit App at: **`http://localhost:8501`**

---

## Running Automated Tests

Run all unit, integration, and API tests:

```bash
cd pdf-rag-gemini
source venv/bin/activate

# 1. Core RAG tests
python test_rag.py

# 2. Phase 1-5 extended tests
python test_extended_phases.py

# 3. FastAPI endpoints tests
python test_api.py
```

Build and test the Angular frontend:
```bash
cd frontend
npm run build
```
