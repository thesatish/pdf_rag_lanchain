import os
import tempfile
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

import rag_pipeline
from rag_pipeline import (
    PERSIST_DIR,
    LLM_MODEL,
    EMBEDDING_MODEL,
    get_chroma_client,
    get_or_create_vectorstore,
    list_all_collections,
    delete_collection,
    list_documents_in_collection,
    ingest_pdf,
    delete_document,
    update_document,
    build_rag_chain,
    retrieve_with_debug,
    transcribe_audio,
    execute_advanced_rag,
    generate_synthetic_qa,
    evaluate_rag_pipeline,
)
from langchain_core.documents import Document

app = FastAPI(
    title="Gemini PDF RAG API",
    description="FastAPI backend exposing full 5-phase RAG pipeline with Gemini and ChromaDB",
    version="1.0.0",
)

# Enable CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class CreateCollectionRequest(BaseModel):
    name: str = Field(..., description="Name of the Chroma collection to create")


class ChatRequest(BaseModel):
    collection_name: str = Field("default_pdf_rag", description="Active Chroma collection")
    query: str = Field(..., description="User query / question")
    mode: str = Field("standard", description="'standard' or 'advanced'")
    use_query_rewriting: bool = True
    use_hybrid: bool = True
    use_reranking: bool = True
    use_compression: bool = True
    k: int = 4


class RetrieveDebugRequest(BaseModel):
    collection_name: str = "default_pdf_rag"
    query: str
    search_type: str = "similarity"  # similarity, mmr, similarity_score_threshold
    k: int = 4
    fetch_k: int = 15
    lambda_mult: float = 0.5
    score_threshold: Optional[float] = None
    metadata_filter: Optional[Dict[str, Any]] = None


class GenerateEvalDatasetRequest(BaseModel):
    collection_name: str = "default_pdf_rag"
    num_questions: int = 3


class RunEvaluationRequest(BaseModel):
    collection_name: str = "default_pdf_rag"
    dataset: List[Dict[str, Any]]
    engine: str = "standard"  # 'standard' or 'advanced'


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/api/health")
def healthcheck():
    """Healthcheck endpoint returning model configuration and system readiness."""
    has_api_key = bool(os.getenv("GOOGLE_API_KEY"))
    return {
        "status": "ok" if has_api_key else "missing_api_key",
        "api_key_configured": has_api_key,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "persist_dir": PERSIST_DIR,
    }


# --- COLLECTIONS ---

@app.get("/api/collections")
def get_collections():
    """List all available collections in ChromaDB with document counts."""
    client = get_chroma_client()
    return list_all_collections(client=client)


@app.post("/api/collections")
def create_collection(req: CreateCollectionRequest):
    """Create or switch to a Chroma collection."""
    clean_name = req.name.strip().replace(" ", "_")
    if not clean_name:
        raise HTTPException(status_code=400, detail="Collection name cannot be empty.")
    client = get_chroma_client()
    col = client.get_or_create_collection(clean_name)
    return {"status": "success", "name": col.name}


@app.delete("/api/collections/{collection_name}")
def remove_collection(collection_name: str):
    """Delete a Chroma collection."""
    client = get_chroma_client()
    success = delete_collection(collection_name, client=client)
    if not success:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
    return {"status": "success", "message": f"Deleted collection '{collection_name}'."}


# --- DOCUMENTS & INGESTION ---

@app.get("/api/documents")
def get_documents(collection_name: str = Query("default_pdf_rag")):
    """List all unique documents and metadata in the specified collection."""
    client = get_chroma_client()
    return list_documents_in_collection(collection_name, client=client)


@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form("default_pdf_rag"),
    prevent_duplicates: bool = Form(True),
):
    """Upload and index one or multiple PDF documents."""
    client = get_chroma_client()
    results = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({
                "status": "error",
                "filename": file.filename,
                "message": "Only PDF files are supported.",
                "chunks_added": 0,
            })
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            res = ingest_pdf(
                pdf_path=tmp_path,
                collection_name=collection_name,
                client=client,
                prevent_duplicates=prevent_duplicates,
                filename=file.filename,
                return_summary=True,
            )
            # Remove vectorstore object from serializable response
            res.pop("vectorstore", None)
            results.append(res)
        except Exception as e:
            results.append({
                "status": "error",
                "filename": file.filename,
                "message": str(e),
                "chunks_added": 0,
            })
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {
        "collection_name": collection_name,
        "total_files": len(files),
        "results": results,
    }


@app.delete("/api/documents/{filename}")
def remove_document(filename: str, collection_name: str = Query("default_pdf_rag")):
    """Delete a document and all its chunks from the collection."""
    client = get_chroma_client()
    res = delete_document(collection_name=collection_name, filename=filename, client=client)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/documents/update")
async def update_doc(
    file: UploadFile = File(...),
    target_filename: str = Form(...),
    collection_name: str = Form("default_pdf_rag"),
):
    """Atomically update an existing document by replacing all its chunks."""
    client = get_chroma_client()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        res = update_document(
            pdf_path=tmp_path,
            collection_name=collection_name,
            filename=target_filename,
            client=client,
        )
        res.pop("vectorstore", None)
        return res
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def handle_rag_exception(e: Exception) -> HTTPException:
    """
    Intelligently maps exceptions to accurate HTTP status codes (401, 429, 400, 503, 500)
    instead of treating every failure as a generic 503.
    """
    if isinstance(e, HTTPException):
        return e

    error_str = str(e).lower()

    # 1. Extract status code if provided directly on the exception object
    raw_code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(raw_code, int) and 400 <= raw_code < 600:
        return HTTPException(status_code=raw_code, detail=str(e))

    # 2. Rate Limiting & Quota Exhaustion (429)
    if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Gemini API quota or rate limit exceeded. Please wait a moment before trying again. ({str(e)})",
        )

    # 3. Authentication & API Key errors (401)
    if "401" in error_str or "api_key" in error_str or "unauthenticated" in error_str or "invalid api key" in error_str:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or missing Gemini API key. Verify your .env configuration. ({str(e)})",
        )

    # 4. Bad Request / Invalid Arguments / Context limits (400)
    if "400" in error_str or "invalid_argument" in error_str or "token limit" in error_str:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request or prompt input. ({str(e)})",
        )

    # 5. Service Overloaded / Temporary Downtime (503)
    if "503" in error_str or "unavailable" in error_str or "high demand" in error_str or "connection" in error_str or "timeout" in error_str:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The AI model service is temporarily unavailable or experiencing high demand. Please try again. ({str(e)})",
        )

    # 6. Fallback internal error (500)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"An unexpected internal error occurred: {str(e)}",
    )


# --- CHAT & Q&A ---

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Answer questions using Standard RAG or Advanced RAG with page citations."""
    client = get_chroma_client()
    vectorstore = get_or_create_vectorstore(collection_name=req.collection_name, client=client)

    try:
        if req.mode == "advanced":
            result = execute_advanced_rag(
                vectorstore=vectorstore,
                query=req.query,
                use_query_rewriting=req.use_query_rewriting,
                use_hybrid=req.use_hybrid,
                use_reranking=req.use_reranking,
                use_compression=req.use_compression,
                top_k=req.k,
            )
            answer = result["answer"]
            docs = result["docs"]
            telemetry = result.get("telemetry", {})
        else:
            rag_chain = build_rag_chain(vectorstore=vectorstore, k=req.k)
            result = rag_chain.invoke(req.query)
            answer = result["answer"]
            docs = result["docs"]
            telemetry = {"mode": "standard", "k": req.k}
    except Exception as e:
        raise handle_rag_exception(e)

    serializable_docs = []
    for d in docs:
        meta = d.metadata or {}
        serializable_docs.append({
            "filename": meta.get("filename", "unknown.pdf"),
            "page": meta.get("page", 0) + 1,
            "chunk_id": meta.get("chunk_id", ""),
            "content": d.page_content,
        })

    return {
        "answer": answer,
        "docs": serializable_docs,
        "telemetry": telemetry,
    }


@app.post("/api/transcribe-audio")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe speech audio file using Gemini multimodal audio model."""
    audio_bytes = await audio.read()
    mime = audio.content_type or "audio/wav"
    try:
        transcript = transcribe_audio(audio_bytes=audio_bytes, mime_type=mime)
        return {"transcript": transcript}
    except Exception as e:
        raise handle_rag_exception(e)


# --- ADVANCED RETRIEVAL & DIAGNOSTICS ---

@app.post("/api/retrieve/debug")
def retrieve_diagnostic(req: RetrieveDebugRequest):
    """Run retrieval diagnostics and return detailed telemetry."""
    client = get_chroma_client()
    vectorstore = get_or_create_vectorstore(collection_name=req.collection_name, client=client)

    try:
        debug_res = retrieve_with_debug(
            vectorstore=vectorstore,
            query=req.query,
            search_type=req.search_type,
            k=req.k,
            fetch_k=req.fetch_k,
            lambda_mult=req.lambda_mult,
            score_threshold=req.score_threshold,
            metadata_filter=req.metadata_filter,
        )
        return debug_res["debug_info"]
    except Exception as e:
        raise handle_rag_exception(e)


# --- RAG EVALUATION ---

@app.post("/api/eval/generate-dataset")
def generate_eval_dataset(req: GenerateEvalDatasetRequest):
    """Generate synthetic QA pairs from document chunks in the collection."""
    client = get_chroma_client()
    vectorstore = get_or_create_vectorstore(collection_name=req.collection_name, client=client)
    col_data = vectorstore.get()

    chunks = [
        Document(page_content=t, metadata=m)
        for t, m in zip(col_data.get("documents", []), col_data.get("metadatas", []))
    ]
    if not chunks:
        raise HTTPException(status_code=400, detail="No indexed chunks available to generate QA pairs.")

    try:
        dataset = generate_synthetic_qa(chunks, num_questions=req.num_questions)
        return {"dataset": dataset}
    except Exception as e:
        raise handle_rag_exception(e)


@app.post("/api/eval/run")
def run_evaluation(req: RunEvaluationRequest):
    """Execute evaluation benchmark across the provided dataset."""
    client = get_chroma_client()
    vectorstore = get_or_create_vectorstore(collection_name=req.collection_name, client=client)

    def invoke_fn(q: str):
        try:
            if req.engine == "advanced":
                res = execute_advanced_rag(vectorstore=vectorstore, query=q, top_k=4)
                return {"answer": res["answer"], "docs": res["docs"]}
            else:
                chain = build_rag_chain(vectorstore=vectorstore, k=4)
                return chain.invoke(q)
        except Exception as e:
            raise handle_rag_exception(e)

    try:
        eval_result = evaluate_rag_pipeline(invoke_fn=invoke_fn, dataset=req.dataset)
        return eval_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Evaluation pipeline failed. ({str(e)})",
        )


# --- CHROMADB EXPLORER & 2D PCA ---

@app.get("/api/explorer")
def explore_collection(collection_name: str = Query("default_pdf_rag")):
    """Return collection chunk metrics, document content, and 2D PCA projection coordinates."""
    client = get_chroma_client()
    try:
        col = client.get_collection(collection_name)
        data = col.get(include=["documents", "metadatas", "embeddings"])
    except Exception:
        return {"ids": [], "total_chunks": 0, "embedding_dim": "N/A", "chunks": [], "pca_points": []}

    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    embeddings = data.get("embeddings", None)

    embedding_dim = "N/A"
    if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
        embedding_dim = len(embeddings[0])

    chunks = []
    for i, c_id in enumerate(ids):
        meta = metas[i] if metas and i < len(metas) else {}
        chunks.append({
            "chunk_id": c_id,
            "filename": meta.get("filename", "unknown.pdf") if isinstance(meta, dict) else "N/A",
            "page": meta.get("page", 0) + 1 if isinstance(meta, dict) else "N/A",
            "length": len(docs[i]) if docs and i < len(docs) else 0,
            "content": docs[i] if docs and i < len(docs) else "",
            "metadata": meta,
        })

    # 2D PCA projection
    pca_points = []
    if embeddings is not None and len(embeddings) > 1 and embeddings[0] is not None:
        try:
            embs_arr = np.array(embeddings)
            embs_c = embs_arr - np.mean(embs_arr, axis=0)
            u, s, _ = np.linalg.svd(embs_c, full_matrices=False)
            coords_2d = u[:, :2] * s[:2]
            for i in range(len(ids)):
                meta = metas[i] if metas and i < len(metas) else {}
                pca_points.append({
                    "chunk_id": ids[i],
                    "x": round(float(coords_2d[i, 0]), 4),
                    "y": round(float(coords_2d[i, 1]), 4),
                    "filename": meta.get("filename", "unknown.pdf") if isinstance(meta, dict) else "N/A",
                    "page": meta.get("page", 0) + 1 if isinstance(meta, dict) else 1,
                })
        except Exception:
            pass

    return {
        "collection_name": collection_name,
        "total_chunks": len(ids),
        "embedding_dim": embedding_dim,
        "chunks": chunks,
        "pca_points": pca_points,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

