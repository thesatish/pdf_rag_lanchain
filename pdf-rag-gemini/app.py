import os
import tempfile
import uuid
import warnings
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

import rag_pipeline
from rag_pipeline import (
    PERSIST_DIR,
    get_chroma_client,
    get_embeddings,
    get_llm,
    ingest_pdf,
    ingest_multiple_pdfs,
    get_or_create_vectorstore,
    list_documents_in_collection,
    delete_document,
    update_document,
    list_all_collections,
    delete_collection,
    build_rag_chain,
    get_retriever,
    retrieve_with_debug,
    format_docs,
    transcribe_audio,
    rewrite_query,
    hybrid_search,
    rerank_documents,
    compress_context,
    execute_advanced_rag,
    generate_synthetic_qa,
    evaluate_rag_pipeline,
)

st.set_page_config(page_title="Gemini RAG System", layout="wide", page_icon="📚")
st.title("📚 Intelligent PDF RAG System (LangChain + Gemini)")

# Verify Google API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing `GOOGLE_API_KEY`. Add it to your `.env` file and restart.")
    st.stop()

client = get_chroma_client()

# ==============================================================================
# SIDEBAR: COLLECTION SELECTOR & MULTI-PDF INGESTION
# ==============================================================================
with st.sidebar:
    st.header("🗄️ Database & Collections")
    all_cols = list_all_collections(client=client)
    col_names = [c["name"] for c in all_cols] if all_cols else []

    if "current_collection" not in st.session_state:
        st.session_state["current_collection"] = col_names[0] if col_names else "default_pdf_rag"

    # Select existing collection (Reusing DB without re-ingesting!)
    selected_col = st.selectbox(
        "Active Chroma Collection",
        options=col_names if col_names else ["default_pdf_rag"],
        index=col_names.index(st.session_state["current_collection"]) if st.session_state["current_collection"] in col_names else 0,
        help="Switch collections or reuse existing databases without re-uploading documents.",
    )
    st.session_state["current_collection"] = selected_col

    # Create new collection
    with st.expander("➕ Create / Manage Collection"):
        new_col_name = st.text_input("New Collection Name", "").strip()
        if st.button("Create Collection") and new_col_name:
            st.session_state["current_collection"] = new_col_name
            st.success(f"Collection '{new_col_name}' ready!")
            st.rerun()

        if col_names and st.button("🗑️ Delete Active Collection", type="secondary"):
            delete_collection(selected_col, client=client)
            st.warning(f"Deleted collection '{selected_col}'.")
            st.session_state["current_collection"] = "default_pdf_rag"
            st.rerun()

    st.divider()

    # Upload multiple documents (Phase 2)
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple documents into the active collection.",
    )

    prevent_dups = st.checkbox("Prevent Duplicate Ingestion", value=True, help="Uses SHA-256 hash checks to prevent re-indexing identical PDFs.")

    if uploaded_files:
        if st.button("📥 Process & Ingest Files", type="primary"):
            temp_paths = []
            file_names = []
            for up_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up_file.getvalue())
                    temp_paths.append(tmp.name)
                    file_names.append(up_file.name)

            try:
                with st.spinner("Indexing documents into ChromaDB..."):
                    results = []
                    for path, name in zip(temp_paths, file_names):
                        res = ingest_pdf(
                            pdf_path=path,
                            collection_name=st.session_state["current_collection"],
                            client=client,
                            prevent_duplicates=prevent_dups,
                            filename=name,
                            return_summary=True,
                        )
                        results.append(res)

                    succ_count = sum(1 for r in results if r["status"] == "success")
                    dup_count = sum(1 for r in results if r["status"] == "duplicate")

                    if succ_count > 0:
                        st.success(f"Successfully indexed {succ_count} document(s)!")
                    if dup_count > 0:
                        st.info(f"Skipped {dup_count} duplicate document(s).")
                    for r in results:
                        if r["status"] == "duplicate":
                            st.caption(f"⚠️ {r['filename']} already indexed (Hash: {r.get('file_hash', '')[:8]}...).")
            finally:
                for p in temp_paths:
                    if os.path.exists(p):
                        os.remove(p)
            st.rerun()

# Connect vectorstore to active collection (DB reuse)
vectorstore = get_or_create_vectorstore(
    collection_name=st.session_state["current_collection"],
    client=client,
)

# Fetch documents in collection
collection_docs = list_documents_in_collection(st.session_state["current_collection"], client=client)
doc_filenames = [d["filename"] for d in collection_docs]

# ==============================================================================
# MAIN TABS (PHASES 1 - 5)
# ==============================================================================
tab_chat, tab_docs, tab_retrieval, tab_eval, tab_db = st.tabs([
    "💬 Chat & Q&A",
    "📁 Document Management (Phase 2)",
    "🔍 Advanced Retrieval & Debugger (Phase 3)",
    "📈 RAG Evaluation (Phase 4)",
    "📊 ChromaDB Explorer",
])

# ==============================================================================
# TAB 1: CHAT & Q&A (PHASE 1, 3, 5)
# ==============================================================================
with tab_chat:
    st.subheader(f"💬 Document Assistant — Collection: `{st.session_state['current_collection']}`")

    # Mode Selector
    c_m1, c_m2 = st.columns([1, 3])
    with c_m1:
        rag_mode = st.radio("RAG Engine Mode", ["Standard RAG", "Advanced RAG (Phase 5)"], horizontal=True)

    with c_m2:
        if rag_mode == "Advanced RAG (Phase 5)":
            c_a1, c_a2, c_a3, c_a4 = st.columns(4)
            with c_a1:
                use_rewrite = st.checkbox("✍️ Query Rewriting", value=True)
            with c_a2:
                use_hybrid = st.checkbox("🔀 Hybrid BM25", value=True)
            with c_a3:
                use_rerank = st.checkbox("🎯 Reranking", value=True)
            with c_a4:
                use_compress = st.checkbox("✂️ Compression", value=True)
        else:
            use_rewrite, use_hybrid, use_rerank, use_compress = False, False, False, False

    if not collection_docs:
        st.info("💡 No documents found in this collection. Upload a PDF using the sidebar to begin asking questions.")
    else:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display Chat History
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("View retrieved source chunks"):
                        for i, doc in enumerate(msg["sources"], start=1):
                            fname = doc.get("filename", "unknown")
                            page = doc.get("page", 0) + 1
                            st.markdown(f"**Chunk {i} — `{fname}` (Page {page})**")
                            st.caption(doc.get("content", ""))

        # Voice & Text Input
        voice_prompt = st.audio_input("🎤 Voice Question (optional)", key="audio_question")
        text_prompt = st.chat_input("Ask a question about your indexed documents...")

        user_question = None
        if voice_prompt:
            with st.spinner("Transcribing audio input..."):
                user_question = transcribe_audio(voice_prompt.getvalue())
                st.caption(f"🎙️ Transcribed: *\"{user_question}\"*")
        elif text_prompt:
            user_question = text_prompt

        if user_question:
            st.session_state["messages"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and formulating answer..."):
                    try:
                        if rag_mode == "Advanced RAG (Phase 5)":
                            adv_result = execute_advanced_rag(
                                vectorstore=vectorstore,
                                query=user_question,
                                use_query_rewriting=use_rewrite,
                                use_hybrid=use_hybrid,
                                use_reranking=use_rerank,
                                use_compression=use_compress,
                                top_k=4,
                            )
                            answer = adv_result["answer"]
                            docs = adv_result["docs"]
                        else:
                            rag_chain = build_rag_chain(vectorstore=vectorstore, k=4)
                            res = rag_chain.invoke(user_question)
                            answer = res["answer"]
                            docs = res["docs"]
                    except Exception as e:
                        answer = f"⚠️ The AI model service is currently unavailable or experiencing high demand. Please try again in a moment. ({str(e)})"
                        docs = []

                    st.markdown(answer)

                    # Expandable source citations
                    sources_data = []
                    with st.expander("📑 View retrieved source chunks"):
                        for i, doc in enumerate(docs, start=1):
                            meta = doc.metadata or {}
                            fname = meta.get("filename", "document.pdf")
                            page = meta.get("page", 0) + 1
                            st.markdown(f"**Chunk {i} — `{fname}` (Page {page})**")
                            st.caption(doc.page_content)
                            sources_data.append({"filename": fname, "page": meta.get("page", 0), "content": doc.page_content})

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources_data,
            })

# ==============================================================================
# TAB 2: DOCUMENT MANAGEMENT (PHASE 2)
# ==============================================================================
with tab_docs:
    st.subheader(f"📁 Document Inventory for Collection: `{st.session_state['current_collection']}`")

    if not collection_docs:
        st.info("No documents indexed yet in this collection. Use the sidebar to upload files.")
    else:
        df_docs = pd.DataFrame(collection_docs)
        display_df = df_docs[["filename", "total_pages", "chunk_count", "file_size", "upload_time", "file_hash"]].copy()
        display_df.columns = ["Filename", "Total Pages", "Total Chunks", "File Size (Bytes)", "Upload Date", "SHA-256 Hash"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()

        # Update or Delete Documents
        c_del, c_upd = st.columns(2)
        with c_del:
            st.markdown("### 🗑️ Delete Document")
            doc_to_delete = st.selectbox("Select document to delete:", doc_filenames, key="delete_doc_select")
            if st.button("Delete Selected Document", type="secondary"):
                del_res = delete_document(
                    collection_name=st.session_state["current_collection"],
                    filename=doc_to_delete,
                    client=client,
                )
                st.success(f"Document `{doc_to_delete}` deleted ({del_res.get('deleted_count', 0)} chunks removed).")
                st.rerun()

        with c_upd:
            st.markdown("### 🔄 Update / Replace Document")
            doc_to_update = st.selectbox("Select document to replace:", doc_filenames, key="update_doc_select")
            update_file = st.file_uploader("Upload replacement PDF", type=["pdf"], key="update_pdf_uploader")
            if st.button("Update Document") and update_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(update_file.getvalue())
                    tmp_path = tmp.name
                try:
                    with st.spinner("Replacing document chunks..."):
                        upd_res = update_document(
                            pdf_path=tmp_path,
                            collection_name=st.session_state["current_collection"],
                            filename=doc_to_update,
                            client=client,
                        )
                        st.success(f"Document `{doc_to_update}` successfully updated! ({upd_res.get('chunks_added', 0)} new chunks added).")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                st.rerun()

# ==============================================================================
# TAB 3: ADVANCED RETRIEVAL & DEBUGGER (PHASE 3)
# ==============================================================================
with tab_retrieval:
    st.subheader("🔍 Advanced Retrieval Configuration & Debugging")
    st.markdown("Experiment with retrieval algorithms, similarity score cutoffs, and metadata filtering to optimize retrieval quality.")

    c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
    with c_cfg1:
        search_type = st.selectbox(
            "Search Algorithm",
            options=["similarity", "mmr", "similarity_score_threshold"],
            help="Similarity: Standard dense vector cosine/L2 search.\nMMR: Maximal Marginal Relevance for diversity.\nThreshold: Strict score cutoff.",
        )
        top_k = st.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=4)

    with c_cfg2:
        fetch_k = 20
        lambda_mult = 0.5
        score_thresh = 0.5
        if search_type == "mmr":
            fetch_k = st.slider("MMR Fetch-K (Candidate Pool)", min_value=top_k, max_value=30, value=15)
            lambda_mult = st.slider("Diversity Factor (0 = Diverse, 1 = Relevant)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        elif search_type == "similarity_score_threshold":
            score_thresh = st.slider("Relevance Score Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    with c_cfg3:
        filter_doc = st.selectbox("Metadata Filter: Document", options=["All Documents"] + doc_filenames)
        meta_filter = {"filename": filter_doc} if filter_doc != "All Documents" else None

    # Debug Test Query
    st.markdown("---")
    st.markdown("### 🧪 Retrieval Diagnostic Tester")
    test_query = st.text_input("Enter test query to diagnose retrieval:", "What are the core technical concepts?")

    if st.button("🚀 Run Retrieval Diagnostic") and test_query:
        with st.spinner("Executing retrieval with telemetry..."):
            debug_result = retrieve_with_debug(
                vectorstore=vectorstore,
                query=test_query,
                search_type=search_type,
                k=top_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                score_threshold=score_thresh if search_type == "similarity_score_threshold" else None,
                metadata_filter=meta_filter,
            )

            debug_info = debug_result["debug_info"]
            items = debug_info["items"]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Retrieval Latency", f"{debug_info['latency_ms']} ms")
            with m2:
                st.metric("Chunks Retrieved", debug_info["num_retrieved"])
            with m3:
                st.metric("Search Algorithm", debug_info["search_type"].upper())
            with m4:
                top_score = items[0]["score"] if items else "N/A"
                st.metric("Top Chunk Score", top_score)

            if items:
                st.markdown("#### Retrieved Chunks Breakdown")
                table_rows = []
                for it in items:
                    table_rows.append({
                        "Rank": it["rank"],
                        "Score": it["score"],
                        "Document": it["filename"],
                        "Page": it["page"],
                        "Chunk ID": it["chunk_id"],
                        "Length (chars)": it["content_length"],
                        "Content Preview": it["content_preview"],
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

                st.markdown("#### Full Chunk Inspection")
                selected_rank = st.selectbox("Select chunk rank to inspect:", [it["rank"] for it in items])
                chosen_item = next(it for it in items if it["rank"] == selected_rank)
                st.code(chosen_item["full_content"], language="text")
                st.caption(f"Metadata: {chosen_item['metadata']}")
            else:
                st.warning("No chunks met the retrieval criteria. Try lowering the score threshold or checking your document filter.")

# ==============================================================================
# TAB 4: RAG EVALUATION (PHASE 4)
# ==============================================================================
with tab_eval:
    st.subheader("📈 RAG Evaluation Benchmark")
    st.markdown("Generate synthetic question-answer benchmarks and evaluate retrieval recall, context relevance, faithfulness, and answer relevance.")

    col_data = vectorstore.get()
    all_stored_chunks = [
        Document(page_content=t, metadata=m)
        for t, m in zip(col_data.get("documents", []), col_data.get("metadatas", []))
    ]

    if not all_stored_chunks:
        st.info("No document chunks available to evaluate. Ingest documents first.")
    else:
        c_e1, c_e2 = st.columns([1, 1])
        with c_e1:
            st.markdown("#### 1. Generate Synthetic Benchmark Dataset")
            num_qa = st.slider("Number of Test QA Pairs to Generate", min_value=2, max_value=8, value=3)
            if st.button("⚡ Generate Synthetic QA Pairs"):
                with st.spinner("Generating factual question-answer pairs with Gemini..."):
                    dataset = generate_synthetic_qa(all_stored_chunks, num_questions=num_qa)
                    st.session_state["eval_dataset"] = dataset
                    st.success(f"Generated {len(dataset)} benchmark question-answer pairs!")

        with c_e2:
            st.markdown("#### 2. Benchmark Dataset Preview")
            eval_data = st.session_state.get("eval_dataset", [])
            if eval_data:
                st.dataframe(pd.DataFrame(eval_data)[["id", "question", "ground_truth_answer", "source_filename", "source_page"]], use_container_width=True, hide_index=True)
            else:
                st.caption("No benchmark dataset active. Click 'Generate Synthetic QA Pairs' above.")

        st.divider()

        if eval_data:
            st.markdown("#### 3. Run Benchmark Evaluation")
            eval_engine = st.radio("Evaluate Engine", ["Standard RAG", "Advanced RAG (Phase 5)"], horizontal=True, key="eval_engine_select")

            if st.button("🚀 Run RAG Quality Evaluation", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def invoke_target(q):
                    if eval_engine == "Advanced RAG (Phase 5)":
                        adv = execute_advanced_rag(vectorstore, q, top_k=4)
                        return {"answer": adv["answer"], "docs": adv["docs"]}
                    else:
                        chain = build_rag_chain(vectorstore, k=4)
                        return chain.invoke(q)

                def update_prog(curr, tot):
                    progress_bar.progress(curr / tot)
                    status_text.caption(f"Evaluating sample {curr} of {tot}...")

                with st.spinner("Evaluating retrieval recall and LLM answer quality..."):
                    eval_results = evaluate_rag_pipeline(
                        invoke_fn=invoke_target,
                        dataset=eval_data,
                        progress_callback=update_prog,
                    )
                    status_text.empty()
                    progress_bar.empty()

                summary = eval_results["summary"]
                st.markdown("### 🏆 Evaluation Summary Scores")
                s1, s2, s3, s4, s5 = st.columns(5)
                with s1:
                    st.metric("Recall @ K", f"{summary.get('mean_recall_at_k', 0.0) * 100:.1f}%")
                with s2:
                    st.metric("Context Relevance", f"{summary.get('mean_context_relevance', 0.0) * 100:.1f}%")
                with s3:
                    st.metric("Faithfulness", f"{summary.get('mean_faithfulness', 0.0) * 100:.1f}%")
                with s4:
                    st.metric("Answer Relevance", f"{summary.get('mean_answer_relevance', 0.0) * 100:.1f}%")
                with s5:
                    st.metric("Token F1", f"{summary.get('mean_token_f1', 0.0) * 100:.1f}%")

                st.markdown("#### Itemized Evaluation Results")
                df_results = pd.DataFrame(eval_results["results"])
                st.dataframe(
                    df_results[["question", "predicted_answer", "recall_at_k", "context_relevance", "faithfulness", "answer_relevance", "token_f1"]],
                    use_container_width=True,
                    hide_index=True,
                )

# ==============================================================================
# TAB 5: CHROMADB EXPLORER
# ==============================================================================
with tab_db:
    st.subheader(f"📊 ChromaDB Vector Store Explorer (`{st.session_state['current_collection']}`)")

    try:
        target_col = client.get_collection(st.session_state["current_collection"])
        col_data = target_col.get(include=["documents", "metadatas", "embeddings"])
    except Exception:
        col_data = {"ids": [], "documents": [], "metadatas": [], "embeddings": None}

    ids = col_data.get("ids", [])
    docs = col_data.get("documents", [])
    metas = col_data.get("metadatas", [])
    embeddings = col_data.get("embeddings", None)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Indexed Chunks", len(ids))
    with c2:
        dim = len(embeddings[0]) if (embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None) else "N/A"
        st.metric("Embedding Dimension", dim)
    with c3:
        st.metric("Active Collection", st.session_state["current_collection"])

    subtab_docs, subtab_pca = st.tabs(["📄 Chunks & Metadata", "🌌 2D Embedding Projection (PCA)"])

    with subtab_docs:
        if ids:
            table_rows = []
            for i, chunk_id in enumerate(ids):
                meta = metas[i] if metas and i < len(metas) else {}
                doc = docs[i] if docs and i < len(docs) else ""
                table_rows.append({
                    "Chunk ID": chunk_id,
                    "Document": meta.get("filename", "unknown.pdf") if isinstance(meta, dict) else "N/A",
                    "Page": meta.get("page", 0) + 1 if isinstance(meta, dict) else "N/A",
                    "Characters": len(doc),
                    "Content Preview": doc[:120] + "..." if len(doc) > 120 else doc,
                    "Full Content": doc,
                })

            df = pd.DataFrame(table_rows)
            kw = st.text_input("🔍 Filter chunks by keyword", "")
            filtered = df[df["Full Content"].str.contains(kw, case=False, na=False)] if kw else df

            st.dataframe(filtered[["Chunk ID", "Document", "Page", "Characters", "Content Preview"]], use_container_width=True, hide_index=True)

            st.markdown("**Inspect Chunk Content**")
            selected_id = st.selectbox("Select Chunk ID:", ids, key="chunk_inspector")
            if selected_id:
                idx = ids.index(selected_id)
                st.caption(f"Metadata: {metas[idx] if metas and idx < len(metas) else {}}")
                st.text_area("Chunk Content", docs[idx], height=180, key="chunk_text_area")
        else:
            st.info("No chunks stored in this collection.")

    with subtab_pca:
        if embeddings is not None and len(embeddings) > 1 and embeddings[0] is not None:
            embs_arr = np.array(embeddings)
            embs_c = embs_arr - np.mean(embs_arr, axis=0)
            u, s, _ = np.linalg.svd(embs_c, full_matrices=False)
            coords_2d = u[:, :2] * s[:2]

            plot_df = pd.DataFrame({
                "x": coords_2d[:, 0],
                "y": coords_2d[:, 1],
                "Chunk ID": ids,
                "Document": [m.get("filename", "unknown.pdf") if isinstance(m, dict) else "N/A" for m in metas],
            })
            st.scatter_chart(plot_df, x="x", y="y", color="Document", use_container_width=True)
            st.caption("2D projection of embedding vectors via PCA. Closer clusters indicate semantic closeness.")
        else:
            st.info("At least 2 embeddings are required for 2D semantic projection.")
