# Extended Multi-Phase PDF RAG System with Gemini, LangChain, FastAPI & Angular

A modular, enterprise-grade Document Question-Answering (RAG) system powered by **LangChain**, **Google Gemini** (`gemini-3.8-flash` & `gemini-embedding-2`), **ChromaDB**, **FastAPI**, **Angular 19**, and **Streamlit**.

Both frontends are fully supported and can be run simultaneously or independently:
1. **Modern Angular SPA (`../frontend`) + FastAPI REST Backend (`server.py`)**
2. **Streamlit Application (`app.py`)**

---

## 5-Phase Learning Roadmap

1. **Phase 1 — Refactored Modular RAG:** Decoupled Ingestion, Retrieval, and Generation modules with persistent database reuse (no re-ingesting required).
2. **Phase 2 — Full Document Management:** Multi-PDF support, batch uploads, document updates/replacements, deletion, metadata enrichment, and SHA-256 duplicate prevention.
3. **Phase 3 — Advanced Retrieval & Debugging:** Maximal Marginal Relevance (MMR), similarity score thresholds, metadata filtering, and interactive telemetry/diagnostics.
4. **Phase 4 — RAG Evaluation:** Automated synthetic benchmark dataset generation (triplets of Question, Context, Ground-Truth Answer), measuring Recall@k, Context Relevance, Faithfulness, Answer Relevance, and Token F1.
5. **Phase 5 — Advanced RAG Techniques:** Multi-query expansion / rewriting, hybrid search (Sparse BM25 + Dense Chroma with Reciprocal Rank Fusion), LLM-based reranking, and contextual compression.

---

## REST API Endpoints in `server.py`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Healthcheck and model configuration readiness |
| `GET` | `/api/collections` | List all available ChromaDB collections and chunk counts |
| `POST` | `/api/collections` | Create or switch active Chroma collection |
| `DELETE` | `/api/collections/{name}` | Delete a Chroma collection |
| `GET` | `/api/documents` | List all unique documents and metadata in collection |
| `POST` | `/api/documents/upload` | Upload & ingest multiple PDFs with SHA-256 duplicate prevention |
| `DELETE` | `/api/documents/{filename}` | Delete all chunks associated with a specific document |
| `POST` | `/api/documents/update` | Atomically replace an existing document with a new PDF version |
| `POST` | `/api/chat` | Chat Q&A with Standard RAG or Advanced RAG (Phase 5) |
| `POST` | `/api/transcribe-audio` | Transcribe speech audio using Gemini multimodal capabilities |
| `POST` | `/api/retrieve/debug` | Retrieval diagnostics: latency in ms, similarity scores, ranks, metadata |
| `POST` | `/api/eval/generate-dataset` | Generate synthetic benchmark QA dataset with Gemini |
| `POST` | `/api/eval/run` | Execute evaluation benchmark (Recall@k, Faithfulness, Relevance, F1) |
| `GET` | `/api/explorer` | Inspect chunks, token lengths, and 2D PCA projection coordinates |

---

## How to Run

### 1. Launch FastAPI Backend
```bash
cd pdf-rag-gemini
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs: **`http://localhost:8000/docs`**

### 2. Launch Angular 19 Frontend
```bash
cd ../frontend
npm start
```
Access at: **`http://localhost:4200`**

### 3. Launch Streamlit App (Optional)
```bash
cd pdf-rag-gemini
source venv/bin/activate
streamlit run app.py
```
Access at: **`http://localhost:8501`**

---

## Running Tests

```bash
cd pdf-rag-gemini
source venv/bin/activate

python test_rag.py              # Core tests (6 passed)
python test_extended_phases.py  # Phase 1-5 tests (15 passed)
python test_api.py              # FastAPI tests (4 passed)
```
