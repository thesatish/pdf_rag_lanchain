import json
import re
from typing import List, Dict, Any, Optional, Callable
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from .config import get_llm
from .generation import format_docs


# ==============================================================================
# 1. TEST DATASET GENERATION
# ==============================================================================

def generate_synthetic_qa(
    docs: List[Document],
    num_questions: int = 5,
    llm=None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic question-answer-context triplets from document chunks.
    Useful for building automated RAG evaluation benchmarks.
    """
    if not docs:
        return []

    if llm is None:
        llm = get_llm(temperature=0.3)

    dataset: List[Dict[str, Any]] = []
    # Pick diverse chunks across documents/pages
    step = max(1, len(docs) // max(1, num_questions))
    selected_chunks = docs[::step][:num_questions]

    for idx, chunk in enumerate(selected_chunks, start=1):
        content = chunk.page_content.strip()
        filename = chunk.metadata.get("filename", "unknown.pdf")
        page = chunk.metadata.get("page", 0) + 1
        chunk_id = chunk.metadata.get("chunk_id", f"c_{idx}")

        prompt = (
            f"You are a strict data scientist creating a golden test set for a RAG evaluation benchmark.\n"
            f"Based on the following document passage ONLY, write ONE clear factual question and its accurate answer.\n\n"
            f"Passage:\n{content[:800]}\n\n"
            f"Output JSON with this exact structure:\n"
            f'{{"question": "...", "ground_truth_answer": "..."}}\n'
            f"Output ONLY valid JSON, with no markdown code blocks or surrounding text."
        )

        try:
            resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            # Clean possible markdown fence
            resp_clean = re.sub(r"^```(json)?|```$", "", resp).strip()
            data = json.loads(resp_clean)
            dataset.append({
                "id": f"eval_{idx}",
                "question": data.get("question", ""),
                "ground_truth_answer": data.get("ground_truth_answer", ""),
                "source_chunk_id": chunk_id,
                "source_filename": filename,
                "source_page": page,
                "source_context": content[:300],
            })
        except Exception:
            # Fallback heuristic question if JSON parsing fails
            words = content.split()[:15]
            summary = " ".join(words)
            dataset.append({
                "id": f"eval_{idx}",
                "question": f"What does the document state regarding {summary}?",
                "ground_truth_answer": content[:200],
                "source_chunk_id": chunk_id,
                "source_filename": filename,
                "source_page": page,
                "source_context": content[:300],
            })

    return dataset


# ==============================================================================
# 2. EVALUATION METRICS
# ==============================================================================

def token_f1_score(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute token-level Precision, Recall, and F1 score between prediction and gold answer."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gold_tokens = re.findall(r"\w+", ground_truth.lower())

    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = collections_counter = {}
    for t in pred_tokens:
        collections_counter[t] = collections_counter.get(t, 0) + 1
    
    overlap = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    for t, cnt in gold_counts.items():
        overlap += min(cnt, collections_counter.get(t, 0))

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def recall_at_k(
    retrieved_docs: List[Document],
    target_chunk_id: Optional[str] = None,
    target_page: Optional[int] = None,
    target_filename: Optional[str] = None,
) -> float:
    """Check if the expected chunk or page was successfully retrieved in top-k."""
    for doc in retrieved_docs:
        meta = doc.metadata or {}
        if target_chunk_id and meta.get("chunk_id") == target_chunk_id:
            return 1.0
        if target_page is not None and (meta.get("page", -1) + 1) == target_page:
            if not target_filename or meta.get("filename") == target_filename:
                return 1.0
    return 0.0


def context_relevance_score(query: str, retrieved_docs: List[Document], llm=None) -> float:
    """
    LLM-as-a-judge metric: assesses how relevant the retrieved context chunks
    are for answering the query (0.0 to 1.0).
    """
    if not retrieved_docs:
        return 0.0
    if llm is None:
        llm = get_llm(temperature=0.0)

    context_str = format_docs(retrieved_docs)
    prompt = (
        f"You are an evaluator assessing retrieval quality.\n"
        f"Query: \"{query}\"\n\n"
        f"Retrieved Context:\n{context_str[:1200]}\n\n"
        f"Score the relevance of this retrieved context for answering the query on a scale from 0.0 to 1.0 "
        f"(1.0 = completely relevant and sufficient, 0.5 = partially relevant, 0.0 = completely irrelevant).\n"
        f"Respond ONLY with a single numeric decimal score between 0.0 and 1.0."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        match = re.search(r"(\d+(\.\d+)?)", resp)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.5
    except Exception:
        return 0.5


def faithfulness_score(answer: str, retrieved_docs: List[Document], llm=None) -> float:
    """
    LLM-as-a-judge metric: assesses whether the generated answer is strictly grounded
    in the context without introducing hallucinated claims (0.0 to 1.0).
    """
    if not retrieved_docs or not answer:
        return 0.0
    if llm is None:
        llm = get_llm(temperature=0.0)

    context_str = format_docs(retrieved_docs)
    prompt = (
        f"You are a strict fact-checking evaluator assessing RAG faithfulness.\n"
        f"Context:\n{context_str[:1200]}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Is every claim made in the answer directly supported by the context?\n"
        f"Score faithfulness on a scale from 0.0 to 1.0 (1.0 = 100% faithful and grounded, 0.0 = completely fabricated).\n"
        f"Respond ONLY with a single numeric decimal score between 0.0 and 1.0."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        match = re.search(r"(\d+(\.\d+)?)", resp)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.5
    except Exception:
        return 0.5


def answer_relevance_score(query: str, answer: str, llm=None) -> float:
    """
    LLM-as-a-judge metric: assesses how directly and completely the generated
    answer addresses the user's question (0.0 to 1.0).
    """
    if not answer or not query:
        return 0.0
    if llm is None:
        llm = get_llm(temperature=0.0)

    prompt = (
        f"You are an evaluator assessing answer relevance.\n"
        f"Question: \"{query}\"\n\n"
        f"Answer:\n{answer}\n\n"
        f"Score how directly and helpfully the answer addresses the question from 0.0 to 1.0 "
        f"(1.0 = completely addresses question, 0.0 = completely off-topic or evasive).\n"
        f"Respond ONLY with a single numeric decimal score between 0.0 and 1.0."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        match = re.search(r"(\d+(\.\d+)?)", resp)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.5
    except Exception:
        return 0.5


# ==============================================================================
# 3. EVALUATION RUNNER
# ==============================================================================

def evaluate_rag_pipeline(
    invoke_fn: Callable[[str], Dict[str, Any]],
    dataset: List[Dict[str, Any]],
    llm=None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Run automated evaluation across a test dataset.
    Computes Recall@k, Context Relevance, Faithfulness, Answer Relevance, and Token F1.
    """
    results = []
    total = len(dataset)

    for i, item in enumerate(dataset):
        query = item["question"]
        gold_answer = item.get("ground_truth_answer", "")
        target_chunk_id = item.get("source_chunk_id")
        target_page = item.get("source_page")
        target_filename = item.get("source_filename")

        # Invoke the RAG chain/pipeline
        res = invoke_fn(query)
        answer = res.get("answer", "")
        docs = res.get("docs", [])

        # Calculate metrics
        rec = recall_at_k(docs, target_chunk_id=target_chunk_id, target_page=target_page, target_filename=target_filename)
        ctx_rel = context_relevance_score(query, docs, llm=llm)
        faith = faithfulness_score(answer, docs, llm=llm)
        ans_rel = answer_relevance_score(query, answer, llm=llm)
        f1_metrics = token_f1_score(answer, gold_answer)

        results.append({
            "id": item.get("id", f"eval_{i+1}"),
            "question": query,
            "ground_truth": gold_answer,
            "predicted_answer": answer,
            "recall_at_k": rec,
            "context_relevance": ctx_rel,
            "faithfulness": faith,
            "answer_relevance": ans_rel,
            "token_f1": f1_metrics["f1"],
            "retrieved_count": len(docs),
        })

        if progress_callback:
            progress_callback(i + 1, total)

    if not results:
        return {"summary": {}, "results": []}

    summary = {
        "mean_recall_at_k": round(sum(r["recall_at_k"] for r in results) / len(results), 4),
        "mean_context_relevance": round(sum(r["context_relevance"] for r in results) / len(results), 4),
        "mean_faithfulness": round(sum(r["faithfulness"] for r in results) / len(results), 4),
        "mean_answer_relevance": round(sum(r["answer_relevance"] for r in results) / len(results), 4),
        "mean_token_f1": round(sum(r["token_f1"] for r in results) / len(results), 4),
        "total_evaluated": len(results),
    }

    return {
        "summary": summary,
        "results": results,
    }

