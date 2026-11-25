from .config import (
    get_chroma_client,
    reset_chroma_client,
    get_embeddings,
    get_llm,
    PERSIST_DIR,
    LLM_MODEL,
    EMBEDDING_MODEL,
)

from .generation import (
    format_docs,
    get_qa_prompt,
    build_generation_chain,
    generate_answer,
    transcribe_audio,
)

from .ingestion import (
    compute_file_hash,
    check_duplicate,
    load_and_chunk_pdf,
    ingest_pdf,
    ingest_multiple_pdfs,
    list_documents_in_collection,
    delete_document,
    update_document,
    get_or_create_vectorstore,
    list_all_collections,
    delete_collection,
)

from .retrieval import (
    get_retriever,
    retrieve_with_debug,
    build_rag_chain,
)

from .advanced_rag import (
    rewrite_query,
    BM25Index,
    reciprocal_rank_fusion,
    hybrid_search,
    rerank_documents,
    compress_context,
    execute_advanced_rag,
)

from .evaluation import (
    generate_synthetic_qa,
    token_f1_score,
    recall_at_k,
    context_relevance_score,
    faithfulness_score,
    answer_relevance_score,
    evaluate_rag_pipeline,
)

__all__ = [
    # Config
    "get_chroma_client",
    "reset_chroma_client",
    "get_embeddings",
    "get_llm",
    "PERSIST_DIR",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    # Generation
    "format_docs",
    "get_qa_prompt",
    "build_generation_chain",
    "generate_answer",
    "transcribe_audio",
    # Ingestion & Document Management
    "compute_file_hash",
    "check_duplicate",
    "load_and_chunk_pdf",
    "ingest_pdf",
    "ingest_multiple_pdfs",
    "list_documents_in_collection",
    "delete_document",
    "update_document",
    "get_or_create_vectorstore",
    "list_all_collections",
    "delete_collection",
    # Retrieval
    "get_retriever",
    "retrieve_with_debug",
    "build_rag_chain",
    # Advanced RAG
    "rewrite_query",
    "BM25Index",
    "reciprocal_rank_fusion",
    "hybrid_search",
    "rerank_documents",
    "compress_context",
    "execute_advanced_rag",
    # Evaluation
    "generate_synthetic_qa",
    "token_f1_score",
    "recall_at_k",
    "context_relevance_score",
    "faithfulness_score",
    "answer_relevance_score",
    "evaluate_rag_pipeline",
]

