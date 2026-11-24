import os
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .config import get_chroma_client, get_embeddings, PERSIST_DIR


def compute_file_hash(file_source: Union[str, bytes]) -> str:
    """Compute SHA-256 hash of a file path or file bytes for duplicate detection."""
    sha256 = hashlib.sha256()
    if isinstance(file_source, bytes):
        sha256.update(file_source)
    else:
        with open(file_source, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
    return sha256.hexdigest()


def get_or_create_vectorstore(
    collection_name: str = "pdf_rag",
    client: Optional[chromadb.PersistentClient] = None,
    embeddings=None,
) -> Chroma:
    """
    Connect to or create a persistent Chroma vectorstore collection.
    Allows reusing an existing database without re-ingesting PDFs.
    """
    if client is None:
        client = get_chroma_client()
    if embeddings is None:
        embeddings = get_embeddings()

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


def check_duplicate(
    collection_name: str,
    file_hash: str,
    client: Optional[chromadb.PersistentClient] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check if a document with the same SHA-256 hash already exists in the collection."""
    if client is None:
        client = get_chroma_client()

    try:
        col = client.get_collection(collection_name)
    except Exception:
        return False, None

    # Query by file_hash metadata
    results = col.get(where={"file_hash": file_hash}, limit=1, include=["metadatas"])
    metadatas = results.get("metadatas", [])
    if metadatas and len(metadatas) > 0:
        return True, metadatas[0]

    return False, None


def load_and_chunk_pdf(
    pdf_path: str,
    filename: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    custom_doc_id: Optional[str] = None,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Load a PDF, compute hash and metadata, and split into overlapping chunks."""
    actual_filename = filename or os.path.basename(pdf_path)
    file_hash = compute_file_hash(pdf_path)
    file_size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
    doc_id = custom_doc_id or f"doc_{uuid.uuid4().hex[:10]}"
    upload_time = datetime.now().isoformat()

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    total_pages = len(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_counter = 0

    for page_doc in pages:
        page_num = page_doc.metadata.get("page", 0)
        page_chunks = splitter.split_text(page_doc.page_content)
        for chunk_text in page_chunks:
            if not chunk_text.strip():
                continue
            chunk_id = f"{doc_id}_c{chunk_counter}"
            chunk_metadata = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "filename": actual_filename,
                "file_hash": file_hash,
                "file_size": file_size,
                "page": page_num,
                "total_pages": total_pages,
                "chunk_index": chunk_counter,
                "upload_time": upload_time,
            }
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
            chunk_counter += 1

    doc_summary = {
        "doc_id": doc_id,
        "filename": actual_filename,
        "file_hash": file_hash,
        "file_size": file_size,
        "total_pages": total_pages,
        "chunk_count": len(chunks),
        "upload_time": upload_time,
    }

    return chunks, doc_summary


def ingest_pdf(
    pdf_path: str,
    collection_name: str = "pdf_rag",
    client: Optional[chromadb.PersistentClient] = None,
    prevent_duplicates: bool = True,
    filename: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    return_summary: bool = False,
    embeddings=None,
) -> Union[Chroma, Dict[str, Any]]:
    """
    Ingest a single PDF with duplicate detection and metadata preservation.
    If return_summary=False (default), returns the Chroma vectorstore directly for full backward compatibility.
    If return_summary=True, returns a dict with status, doc metadata, chunk counts, and vectorstore.
    """
    if client is None:
        client = get_chroma_client()
    if embeddings is None:
        embeddings = get_embeddings()

    actual_filename = filename or os.path.basename(pdf_path)
    file_hash = compute_file_hash(pdf_path)

    # Check for duplicate
    if prevent_duplicates:
        is_dup, existing_meta = check_duplicate(collection_name, file_hash, client=client)
        if is_dup and existing_meta:
            vectorstore = get_or_create_vectorstore(collection_name=collection_name, client=client, embeddings=embeddings)
            if not return_summary:
                return vectorstore
            return {
                "status": "duplicate",
                "message": f"Duplicate document detected! '{actual_filename}' already indexed as '{existing_meta.get('filename')}' (Hash: {file_hash[:8]}...).",
                "doc_id": existing_meta.get("doc_id"),
                "filename": existing_meta.get("filename"),
                "file_hash": file_hash,
                "chunks_added": 0,
                "vectorstore": vectorstore,
            }

    # Load and split
    chunks, doc_summary = load_and_chunk_pdf(
        pdf_path=pdf_path,
        filename=actual_filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        vectorstore = get_or_create_vectorstore(collection_name=collection_name, client=client, embeddings=embeddings)
        if not return_summary:
            return vectorstore
        return {
            "status": "empty",
            "message": f"No extractable text found in '{actual_filename}'.",
            "chunks_added": 0,
            "vectorstore": vectorstore,
            **doc_summary,
        }

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
        client=client,
        ids=[c.metadata["chunk_id"] for c in chunks],
    )

    if not return_summary:
        return vectorstore

    return {
        "status": "success",
        "message": f"Successfully indexed '{actual_filename}' ({len(chunks)} chunks across {doc_summary['total_pages']} pages).",
        "vectorstore": vectorstore,
        "chunks_added": len(chunks),
        **doc_summary,
    }


def ingest_multiple_pdfs(
    pdf_paths: List[str],
    collection_name: str = "pdf_rag",
    client: Optional[chromadb.PersistentClient] = None,
    prevent_duplicates: bool = True,
) -> Dict[str, Any]:
    """Ingest multiple PDF files sequentially into the same collection."""
    if client is None:
        client = get_chroma_client()

    results = []
    total_added_chunks = 0
    duplicate_count = 0

    for path in pdf_paths:
        res = ingest_pdf(
            pdf_path=path,
            collection_name=collection_name,
            client=client,
            prevent_duplicates=prevent_duplicates,
        )
        results.append(res)
        if res["status"] == "success":
            total_added_chunks += res.get("chunks_added", 0)
        elif res["status"] == "duplicate":
            duplicate_count += 1

    vectorstore = get_or_create_vectorstore(collection_name=collection_name, client=client)
    return {
        "total_files": len(pdf_paths),
        "successful_ingestions": len([r for r in results if r["status"] == "success"]),
        "duplicates_skipped": duplicate_count,
        "total_chunks_added": total_added_chunks,
        "file_results": results,
        "vectorstore": vectorstore,
    }


def list_documents_in_collection(
    collection_name: str,
    client: Optional[chromadb.PersistentClient] = None,
) -> List[Dict[str, Any]]:
    """List all unique documents and metadata stored in a collection."""
    if client is None:
        client = get_chroma_client()

    try:
        col = client.get_collection(collection_name)
    except Exception:
        return []

    data = col.get(include=["metadatas"])
    metas = data.get("metadatas", [])
    if not metas:
        return []

    docs_map: Dict[str, Dict[str, Any]] = {}
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        doc_id = meta.get("doc_id") or meta.get("filename") or "unknown"
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown.pdf"),
                "file_hash": meta.get("file_hash", "N/A"),
                "file_size": meta.get("file_size", 0),
                "total_pages": meta.get("total_pages", 0),
                "upload_time": meta.get("upload_time", "N/A"),
                "chunk_count": 0,
            }
        docs_map[doc_id]["chunk_count"] += 1
        # Track max page seen if total_pages wasn't saved
        page_val = meta.get("page", 0) + 1
        if page_val > docs_map[doc_id]["total_pages"]:
            docs_map[doc_id]["total_pages"] = page_val

    return list(docs_map.values())


def delete_document(
    collection_name: str,
    doc_id: Optional[str] = None,
    filename: Optional[str] = None,
    client: Optional[chromadb.PersistentClient] = None,
) -> Dict[str, Any]:
    """Delete all chunks for a specific document from a collection."""
    if client is None:
        client = get_chroma_client()

    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        return {"status": "error", "message": f"Collection '{collection_name}' not found: {str(e)}"}

    where_filter = {}
    if doc_id:
        where_filter = {"doc_id": doc_id}
    elif filename:
        where_filter = {"filename": filename}
    else:
        return {"status": "error", "message": "Must provide either doc_id or filename to delete."}

    matching = col.get(where=where_filter)
    ids_to_delete = matching.get("ids", [])
    if not ids_to_delete:
        return {"status": "not_found", "message": "No matching document chunks found to delete.", "deleted_count": 0}

    col.delete(ids=ids_to_delete)
    return {
        "status": "success",
        "message": f"Deleted {len(ids_to_delete)} chunks for document.",
        "deleted_count": len(ids_to_delete),
    }


def update_document(
    pdf_path: str,
    collection_name: str,
    filename: Optional[str] = None,
    client: Optional[chromadb.PersistentClient] = None,
) -> Dict[str, Any]:
    """
    Update an existing document by deleting old chunks and re-indexing the new PDF content.
    """
    if client is None:
        client = get_chroma_client()

    target_name = filename or os.path.basename(pdf_path)

    # 1. Delete previous version if present
    delete_res = delete_document(collection_name, filename=target_name, client=client)

    # 2. Ingest updated version
    ingest_res = ingest_pdf(
        pdf_path=pdf_path,
        collection_name=collection_name,
        client=client,
        prevent_duplicates=False,  # Allow replacement
        filename=target_name,
    )
    ingest_res["previous_deleted_chunks"] = delete_res.get("deleted_count", 0)
    ingest_res["status"] = "updated" if ingest_res["status"] == "success" else ingest_res["status"]
    return ingest_res


def list_all_collections(client: Optional[chromadb.PersistentClient] = None) -> List[Dict[str, Any]]:
    """List all available collections in ChromaDB with chunk counts."""
    if client is None:
        client = get_chroma_client()

    collections = client.list_collections()
    summary = []
    for col in collections:
        try:
            count = col.count()
        except Exception:
            count = len(col.get().get("ids", []))
        summary.append({
            "name": col.name,
            "count": count,
            "id": str(col.id) if hasattr(col, "id") else "",
        })
    return summary


def delete_collection(collection_name: str, client: Optional[chromadb.PersistentClient] = None) -> bool:
    """Delete an entire Chroma collection."""
    if client is None:
        client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False
