# PDF Q&A with Gemini RAG

A Document Question-Answering (RAG) AI Agent powered by **LangChain**, **Google Gemini** (`gemini-3.8-flash` & `gemini-embedding-2`), **ChromaDB**, and **Streamlit**.

Upload any PDF document, automatically chunk and index it into a vector store, and ask contextual questions. The assistant answers strictly using the document context and cites exact page numbers.

---

## Features

- **Document Ingestion & Chunking:** Loads multi-page PDFs using `PyPDFLoader` and splits them into overlapping chunks using `RecursiveCharacterTextSplitter`.
- **Gemini Embeddings:** Generates vector embeddings using `gemini-embedding-2`.
- **ChromaDB Vector Store:** Persists indexed documents locally for fast similarity retrieval.
- **LCEL RAG Pipeline:** Composable LangChain Expression Language (LCEL) chain combining similarity search, system prompt formatting with page citations, and `gemini-3.8-flash`.
- **Interactive Streamlit Interface:**
  - **💬 Chat Q&A:** Chatbot UI displaying answers and expandable source context cards with page references.
  - **📊 ChromaDB Explorer:** Inspect indexed collections, search chunks, view metadata, and visualize high-dimensional embeddings in **2D PCA space**.
- **Configurable Environment:** Centralized control over API keys, models, and database paths via `.env`.

---

## Project Structure

```text
pdf-rag-gemini/
├── .env                # Google Gemini API key and model configurations
├── requirements.txt    # Python package dependencies
├── rag_pipeline.py     # Core LangChain PDF ingestion & LCEL RAG logic
├── app.py              # Streamlit web interface (Chat + ChromaDB Explorer)
├── visualize_db.py     # Standalone ChromaDB visualizer dashboard
├── sample.pdf          # Sample multi-page PDF for testing
├── test_rag.py         # Automated test suite (unit & integration tests)
└── chroma_db/          # Persistent local Chroma vector database
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+ (Python 3.10+ recommended)
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Create and Activate Virtual Environment

```bash
cd pdf-rag-gemini

# Create virtual environment
python3 -m venv venv

# Activate on macOS / Linux:
source venv/bin/activate

# (On Windows):
# .\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

Edit the `.env` file in the `pdf-rag-gemini/` directory:

```env
# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Model Configurations & Persistence
EMBEDDING_MODEL=gemini-embedding-2
LLM_MODEL=gemini-3.8-flash
PERSIST_DIR=./chroma_db
```

---

## Running the Application

### Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

### Usage
1. In the sidebar, click **"Choose a PDF file"** and upload your document (or test with `sample.pdf`).
2. Wait for the status indicator: `Indexed: <your-document>.pdf`.
3. In the **"💬 Chat Q&A"** tab, ask any question about your document.
4. Click **"View retrieved source chunks"** under any answer to inspect the exact context and page numbers used by the model.

---

## Visualizing ChromaDB Vector Store

You have multiple options to visualize the data stored in ChromaDB:

### 1. Built-in Explorer (In `app.py`)
Navigate to the **"📊 ChromaDB Explorer"** tab in your running Streamlit app:
- **Metrics:** Total chunks, embedding dimensions, and active collection name.
- **Searchable Table:** Filter chunks by keyword, view character counts and page numbers.
- **2D PCA Semantic Projection:** Interactive scatter plot showing how chunks cluster in vector space.

### 2. Standalone Visualizer Dashboard
Run the dedicated database visualizer on a separate port:

```bash
streamlit run visualize_db.py --server.port 8502
```
Access at **`http://localhost:8502`**.

### 3. Direct SQLite Database Inspection
ChromaDB persists metadata and document content in `chroma_db/chroma.sqlite3`:
- **VS Code:** Install the **SQLite Viewer** extension and click `chroma_db/chroma.sqlite3`.
- **Desktop GUI:** Open with [DB Browser for SQLite](https://sqlitebrowser.org/).

---

## Running Tests

An automated test suite is included to verify all components (document loading, chunking, Chroma indexing, RAG chain execution, and Streamlit compilation):

```bash
python test_rag.py
```
