import time
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from .generation import build_generation_chain, format_docs
from .config import get_llm


def get_retriever(
    vectorstore: Chroma,
    search_type: str = "similarity",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
):
    """
    Build a configurable LangChain retriever supporting:
    - 'similarity': standard vector similarity
    - 'mmr': Maximal Marginal Relevance (balances relevance and diversity)
    - 'similarity_score_threshold': strict relevance cutoff
    with optional metadata filtering.
    """
    search_kwargs: Dict[str, Any] = {"k": k}

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    if search_type == "mmr":
        search_kwargs["fetch_k"] = max(fetch_k, k)
        search_kwargs["lambda_mult"] = max(0.0, min(1.0, lambda_mult))
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold if score_threshold is not None else 0.5
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs,
        )

    else:
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )


def retrieve_with_debug(
    vectorstore: Chroma,
    query: str,
    search_type: str = "similarity",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute retrieval and collect deep telemetry:
    - Latency (ms)
    - Similarity scores and distances
    - Chunks with source filenames, pages, and metadata breakdown
    """
    start_time = time.perf_counter()

    docs_with_scores: List[Tuple[Document, float]] = []

    try:
        if search_type == "similarity_score_threshold":
            threshold = score_threshold if score_threshold is not None else 0.5
            raw_results = vectorstore.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter=metadata_filter,
                score_threshold=threshold,
            )
            docs_with_scores = raw_results

        elif search_type == "mmr":
            # Chroma max_marginal_relevance_search
            mmr_docs = vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=max(fetch_k, k),
                lambda_mult=lambda_mult,
                filter=metadata_filter,
            )
            # Fetch similarity scores for the retrieved MMR documents
            score_map = {}
            try:
                scored = vectorstore.similarity_search_with_score(query, k=max(fetch_k, k), filter=metadata_filter)
                for doc, score in scored:
                    c_id = doc.metadata.get("chunk_id", doc.page_content[:50])
                    score_map[c_id] = score
            except Exception:
                pass

            for doc in mmr_docs:
                c_id = doc.metadata.get("chunk_id", doc.page_content[:50])
                docs_with_scores.append((doc, score_map.get(c_id, 0.0)))

        else:
            # Standard similarity search with score
            docs_with_scores = vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=metadata_filter,
            )
    except Exception as e:
        # Fallback to standard search if similarity_with_score encounters embedding issues
        raw_docs = vectorstore.similarity_search(query, k=k, filter=metadata_filter)
        docs_with_scores = [(d, 0.0) for d in raw_docs]

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    docs = [d for d, _ in docs_with_scores]
    debug_items = []

    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        # Convert distance to normalized similarity score if Chroma returns L2 distance
        # For L2: smaller distance means higher similarity
        is_distance = score > 1.0 or score < 0.0
        normalized_score = round(1.0 / (1.0 + score), 4) if is_distance else round(score, 4)

        debug_items.append({
            "rank": idx,
            "chunk_id": doc.metadata.get("chunk_id", f"chunk_{idx}"),
            "score": normalized_score,
            "raw_score": round(score, 4),
            "filename": doc.metadata.get("filename", "unknown.pdf"),
            "page": doc.metadata.get("page", 0) + 1,
            "total_pages": doc.metadata.get("total_pages", "N/A"),
            "content_length": len(doc.page_content),
            "content_preview": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            "full_content": doc.page_content,
            "metadata": doc.metadata,
        })

    return {
        "docs": docs,
        "debug_info": {
            "query": query,
            "search_type": search_type,
            "k": k,
            "fetch_k": fetch_k if search_type == "mmr" else None,
            "lambda_mult": lambda_mult if search_type == "mmr" else None,
            "score_threshold": score_threshold if search_type == "similarity_score_threshold" else None,
            "filter": metadata_filter,
            "latency_ms": elapsed_ms,
            "num_retrieved": len(docs),
            "items": debug_items,
        },
    }


def build_rag_chain(
    vectorstore: Chroma,
    retriever=None,
    search_type: str = "similarity",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    llm=None,
):
    """
    Build a complete LCEL RAG chain that retrieves context and generates an answer.
    """
    if retriever is None:
        retriever = get_retriever(
            vectorstore=vectorstore,
            search_type=search_type,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

    if llm is None:
        llm = get_llm()

    answer_chain = build_generation_chain(llm=llm)

    rag_chain = RunnableParallel(
        {"docs": retriever, "question": RunnablePassthrough()}
    ).assign(answer=answer_chain)

    return rag_chain

