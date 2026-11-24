import math
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_llm
from .retrieval import get_retriever, retrieve_with_debug
from .generation import format_docs, build_generation_chain


# ==============================================================================
# 1. QUERY REWRITING & EXPANSION
# ==============================================================================

def rewrite_query(
    query: str,
    mode: str = "expand",
    llm=None,
) -> List[str]:
    """
    Rewrite or expand a query using Gemini LLM.
    Modes:
    - 'expand': Generates 3 alternative phrasing/keyword variations.
    - 'standalone': De-references pronouns or conversational context into a standalone query.
    - 'hyde': Hypothetical Document Embeddings - generates a hypothetical answer passage.
    """
    if llm is None:
        llm = get_llm(temperature=0.3)

    if mode == "expand":
        prompt = (
            f"You are an AI search assistant. Provide 3 diverse search queries that capture different "
            f"perspectives, synonyms, and keywords for the following question:\n"
            f"Question: \"{query}\"\n\n"
            f"Return ONLY the 3 queries, one per line, with no numbering, prefixes, or bullet points."
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            queries = [line.strip().lstrip("1234567890.- ") for line in resp.split("\n") if line.strip()]
            return queries[:3] or [query]
        except Exception:
            return [query]

    elif mode == "standalone":
        prompt = (
            f"Rewrite the following user query to be a fully self-contained, standalone question "
            f"that can be looked up in a technical document without conversational history:\n"
            f"Query: \"{query}\"\n\n"
            f"Return ONLY the rewritten query text."
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            return [resp] if resp else [query]
        except Exception:
            return [query]

    elif mode == "hyde":
        prompt = (
            f"Write a short, authoritative paragraph (approx 50-80 words) that directly answers "
            f"the following question based on technical documentation:\n"
            f"Question: \"{query}\"\n\n"
            f"Answer:"
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            return [resp] if resp else [query]
        except Exception:
            return [query]

    return [query]


# ==============================================================================
# 2. HYBRID SEARCH: PURE-PYTHON BM25 + RECIPROCAL RANK FUSION (RRF)
# ==============================================================================

def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer with lowercasing."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """
    Lightweight, pure-Python Okapi BM25 implementation for lexical keyword search.
    Requires no external C compiler or specialized packages.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.term_freqs: List[Counter] = []
        self.doc_freqs: Counter = Counter()
        self.corpus_size: int = 0

    def index_documents(self, documents: List[Document]):
        """Index a list of Document objects."""
        self.documents = documents
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        total_length = 0
        for doc in documents:
            tokens = _tokenize(doc.page_content)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            tf = Counter(tokens)
            self.term_freqs.append(tf)

            for token in set(tokens):
                self.doc_freqs[token] += 1

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Compute BM25 scores for the query and return top-k ranked documents."""
        if self.corpus_size == 0:
            return []

        tokens = _tokenize(query)
        scores = [0.0] * self.corpus_size

        for token in tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue

            # Standard Robertson-Spärck Jones IDF
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            for i in range(self.corpus_size):
                tf = self.term_freqs[i].get(token, 0)
                doc_len = self.doc_lengths[i]
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                if denom > 0:
                    scores[i] += idf * (tf * (self.k1 + 1)) / denom

        ranked_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)
        results = [(self.documents[i], scores[i]) for i in ranked_indices[:top_k] if scores[i] > 0]
        return results


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
    top_n: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to merge multiple ranked lists.
    RRF Score(d) = Sum( 1 / (k + rank_i(d)) )
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            doc_key = doc.metadata.get("chunk_id", doc.page_content[:100])
            doc_map[doc_key] = doc
            scores[doc_key] = scores.get(doc_key, 0.0) + (1.0 / (k + rank))

    sorted_keys = sorted(scores.keys(), key=lambda k_id: scores[k_id], reverse=True)
    return [(doc_map[k_id], round(scores[k_id], 6)) for k_id in sorted_keys[:top_n]]


def hybrid_search(
    vectorstore: Chroma,
    query: str,
    all_collection_docs: Optional[List[Document]] = None,
    top_k: int = 4,
    k_fusion: int = 60,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Execute Hybrid Search combining Dense vector search and Sparse BM25 keyword search.
    """
    # 1. Dense vector search
    dense_docs = vectorstore.similarity_search(query, k=top_k * 2)

    # 2. Sparse BM25 search
    # If all collection docs not provided, fetch them from collection
    if all_collection_docs is None:
        try:
            col_data = vectorstore.get()
            docs_text = col_data.get("documents", [])
            metas = col_data.get("metadatas", [])
            all_collection_docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(docs_text, metas)
            ]
        except Exception:
            all_collection_docs = dense_docs

    bm25 = BM25Index()
    bm25.index_documents(all_collection_docs)
    bm25_results = bm25.search(query, top_k=top_k * 2)
    sparse_docs = [doc for doc, _ in bm25_results]

    # 3. Fuse with RRF
    fused_with_scores = reciprocal_rank_fusion([dense_docs, sparse_docs], k=k_fusion, top_n=top_k)
    fused_docs = [d for d, _ in fused_with_scores]

    debug_info = {
        "dense_count": len(dense_docs),
        "sparse_count": len(sparse_docs),
        "fused_count": len(fused_docs),
        "top_fused_scores": [s for _, s in fused_with_scores],
    }
    return fused_docs, debug_info


# ==============================================================================
# 3. RERANKING
# ==============================================================================

def rerank_documents(
    query: str,
    docs: List[Document],
    top_n: int = 4,
    llm=None,
) -> List[Tuple[Document, float]]:
    """
    Rerank candidate documents using LLM-based cross-scoring.
    Scores relevance of each candidate chunk on a 0-10 scale.
    """
    if not docs:
        return []
    if len(docs) <= top_n:
        return [(doc, 10.0 - i) for i, doc in enumerate(docs)]

    if llm is None:
        llm = get_llm(temperature=0.0)

    # Prepare batch prompt
    items_text = []
    for i, doc in enumerate(docs):
        preview = doc.page_content.replace("\n", " ")[:300]
        items_text.append(f"[{i}] {preview}")
    candidates_formatted = "\n\n".join(items_text)

    prompt = (
        f"You are an expert retrieval reranker. Score each document chunk below on a scale from 0 to 10 "
        f"based on how directly and accurately it answers the query.\n\n"
        f"Query: \"{query}\"\n\n"
        f"Candidates:\n{candidates_formatted}\n\n"
        f"Output ONLY one line per candidate in the format 'index: score' (e.g. '0: 8.5'), with no extra text."
    )

    scored_docs: List[Tuple[Document, float]] = []
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        scores = {}
        for line in response.split("\n"):
            match = re.search(r"(\d+)\s*:\s*([\d\.]+)", line)
            if match:
                idx = int(match.group(1))
                score = float(match.group(2))
                scores[idx] = score

        for i, doc in enumerate(docs):
            scored_docs.append((doc, scores.get(i, 0.0)))
    except Exception:
        # Fallback to original order
        scored_docs = [(doc, 10.0 - i) for i, doc in enumerate(docs)]

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_n]


# ==============================================================================
# 4. CONTEXTUAL COMPRESSION
# ==============================================================================

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


def compress_context(
    query: str,
    docs: List[Document],
    max_sentences: int = 4,
    overlap_threshold: float = 0.05,
) -> List[Document]:
    """
    Contextual compression: filters each chunk down to sentences with direct
    lexical and semantic relevance to the query, stripping irrelevant filler and stopwords.
    """
    raw_query_tokens = set(_tokenize(query))
    content_query_tokens = raw_query_tokens - STOPWORDS
    query_tokens = content_query_tokens if content_query_tokens else raw_query_tokens
    compressed_docs = []

    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content.strip())
        if len(sentences) <= max_sentences:
            compressed_docs.append(doc)
            continue

        scored_sentences = []
        for s in sentences:
            raw_s_tokens = set(_tokenize(s))
            if not raw_s_tokens:
                continue
            content_s_tokens = raw_s_tokens - STOPWORDS
            s_tokens = content_s_tokens if content_s_tokens else raw_s_tokens

            # Jaccard overlap of meaningful content tokens with query
            overlap = len(query_tokens & s_tokens) / (len(query_tokens | s_tokens) + 1e-6)
            scored_sentences.append((s, overlap))

        # Sort by relevance, pick top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored_sentences[:max_sentences]
        # Keep original chronological sentence order
        top_text_set = {s for s, _ in top_sentences}
        retained = [s for s in sentences if s in top_text_set]

        compressed_text = " ".join(retained)
        compressed_doc = Document(
            page_content=compressed_text if compressed_text else doc.page_content,
            metadata={**doc.metadata, "original_length": len(doc.page_content), "compressed": True},
        )
        compressed_docs.append(compressed_doc)

    return compressed_docs


# ==============================================================================
# 5. FULL ADVANCED RAG PIPELINE
# ==============================================================================

def execute_advanced_rag(
    vectorstore: Chroma,
    query: str,
    use_query_rewriting: bool = True,
    use_hybrid: bool = True,
    use_reranking: bool = True,
    use_compression: bool = True,
    top_k: int = 4,
    llm=None,
) -> Dict[str, Any]:
    """
    Run an end-to-end Advanced RAG pipeline combining:
    Query Rewriting -> Hybrid Search / Dense Retrieval -> Reranking -> Compression -> Generation
    """
    if llm is None:
        llm = get_llm()

    telemetry = {}

    # Step 1: Query Rewriting
    search_queries = [query]
    if use_query_rewriting:
        expanded = rewrite_query(query, mode="expand", llm=llm)
        search_queries = list(set([query] + expanded))
        telemetry["rewritten_queries"] = search_queries

    # Step 2: Multi-query / Hybrid Retrieval
    candidate_docs: List[Document] = []
    seen_ids = set()

    for q in search_queries:
        if use_hybrid:
            q_docs, _ = hybrid_search(vectorstore, q, top_k=top_k * 2)
        else:
            q_docs = vectorstore.similarity_search(q, k=top_k * 2)

        for d in q_docs:
            cid = d.metadata.get("chunk_id", d.page_content[:80])
            if cid not in seen_ids:
                seen_ids.add(cid)
                candidate_docs.append(d)

    telemetry["initial_candidate_count"] = len(candidate_docs)

    # Step 3: Reranking
    if use_reranking and len(candidate_docs) > top_k:
        reranked = rerank_documents(query, candidate_docs, top_n=top_k, llm=llm)
        selected_docs = [d for d, _ in reranked]
        telemetry["reranked_scores"] = [s for _, s in reranked]
    else:
        selected_docs = candidate_docs[:top_k]

    telemetry["selected_doc_count"] = len(selected_docs)

    # Step 4: Contextual Compression
    if use_compression:
        final_docs = compress_context(query, selected_docs, max_sentences=4)
        telemetry["context_compressed"] = True
    else:
        final_docs = selected_docs

    # Step 5: Answer Generation
    generation_chain = build_generation_chain(llm=llm)
    answer = generation_chain.invoke({"docs": final_docs, "question": query})

    return {
        "answer": answer,
        "docs": final_docs,
        "telemetry": telemetry,
    }
