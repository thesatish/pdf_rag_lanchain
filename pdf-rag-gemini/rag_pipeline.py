"""
RAG Pipeline Orchestrator.
Exposes backward-compatible interfaces for ingestion, retrieval, and generation,
alongside modular Phase 1 - Phase 5 capabilities.
"""
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

import rag_modules
from rag_modules import (
    PERSIST_DIR,
    LLM_MODEL,
    EMBEDDING_MODEL,
    get_chroma_client,
    reset_chroma_client,
    format_docs,
    get_qa_prompt,
    build_generation_chain,
    generate_answer,
    compute_file_hash,
    check_duplicate,
    load_and_chunk_pdf,
    ingest_multiple_pdfs,
    list_documents_in_collection,
    delete_document,
    update_document,
    get_or_create_vectorstore,
    list_all_collections,
    delete_collection,
    get_retriever,
    retrieve_with_debug,
    rewrite_query,
    BM25Index,
    reciprocal_rank_fusion,
    hybrid_search,
    rerank_documents,
    compress_context,
    execute_advanced_rag,
    generate_synthetic_qa,
    token_f1_score,
    recall_at_k,
    context_relevance_score,
    faithfulness_score,
    answer_relevance_score,
    evaluate_rag_pipeline,
)


def get_embeddings(model_name=None):
    """Initialize Gemini embedding model."""
    return rag_modules.config.get_embeddings(model_name)


def get_llm(model_name=None, temperature=0.2, max_retries=2):
    """Initialize Gemini Chat LLM."""
    mod = sys.modules[__name__]
    chat_cls = getattr(mod, "ChatGoogleGenerativeAI", ChatGoogleGenerativeAI)
    return chat_cls(
        model=model_name or LLM_MODEL,
        temperature=temperature,
        max_retries=max_retries,
    )


def ingest_pdf(
    pdf_path: str,
    collection_name: str = "pdf_rag",
    client=None,
    prevent_duplicates: bool = True,
    filename=None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    return_summary: bool = False,
):
    """Load a PDF, split into chunks, embed, and store in ChromaDB."""
    mod = sys.modules[__name__]
    emb = getattr(mod, "get_embeddings")()
    return rag_modules.ingestion.ingest_pdf(
        pdf_path=pdf_path,
        collection_name=collection_name,
        client=client,
        prevent_duplicates=prevent_duplicates,
        filename=filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        return_summary=return_summary,
        embeddings=emb,
    )


def build_rag_chain(
    vectorstore,
    retriever=None,
    search_type: str = "similarity",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold=None,
    metadata_filter=None,
    llm=None,
):
    """Build an LCEL RAG chain that returns both the answer and source chunks."""
    if llm is None:
        llm = get_llm(temperature=0.2)
    return rag_modules.retrieval.build_rag_chain(
        vectorstore=vectorstore,
        retriever=retriever,
        search_type=search_type,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        score_threshold=score_threshold,
        metadata_filter=metadata_filter,
        llm=llm,
    )


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav", llm=None):
    """Transcribe speech audio into text using Gemini's native multimodal capabilities."""
    if llm is None:
        llm = get_llm(temperature=0.0)
    return rag_modules.generation.transcribe_audio(audio_bytes=audio_bytes, mime_type=mime_type, llm=llm)


__all__ = [
    # Constants & Clients
    "PERSIST_DIR",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "get_chroma_client",
    "reset_chroma_client",
    "get_embeddings",
    "get_llm",
    # LangChain classes for patching compatibility
    "GoogleGenerativeAIEmbeddings",
    "ChatGoogleGenerativeAI",
    "Chroma",
    "PyPDFLoader",
    "RecursiveCharacterTextSplitter",
    "ChatPromptTemplate",
    "StrOutputParser",
    "RunnableParallel",
    "RunnablePassthrough",
    # Core RAG
    "ingest_pdf",
    "format_docs",
    "build_rag_chain",
    "transcribe_audio",
    # Document Management (Phase 2)
    "compute_file_hash",
    "check_duplicate",
    "load_and_chunk_pdf",
    "ingest_multiple_pdfs",
    "list_documents_in_collection",
    "delete_document",
    "update_document",
    "get_or_create_vectorstore",
    "list_all_collections",
    "delete_collection",
    # Advanced Retrieval (Phase 3)
    "get_retriever",
    "retrieve_with_debug",
    # Evaluation (Phase 4)
    "generate_synthetic_qa",
    "token_f1_score",
    "recall_at_k",
    "context_relevance_score",
    "faithfulness_score",
    "answer_relevance_score",
    "evaluate_rag_pipeline",
    # Advanced RAG (Phase 5)
    "rewrite_query",
    "BM25Index",
    "reciprocal_rank_fusion",
    "hybrid_search",
    "rerank_documents",
    "compress_context",
    "execute_advanced_rag",
]
