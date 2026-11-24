import os
import tempfile
import uuid
import warnings

# Suppress Python 3.9 deprecation warnings from google-auth/google-api-core
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import chromadb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import ingest_pdf, build_rag_chain, transcribe_audio, get_chroma_client

PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")

st.set_page_config(page_title="PDF Q&A with Gemini RAG", layout="wide")
st.title("PDF Q&A (LangChain + Gemini RAG)")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GOOGLE_API_KEY. Add it to your .env file and restart.")
    st.stop()

# Sidebar: PDF upload & indexing
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("indexed_file_id") != file_id:
            with st.spinner("Reading, chunking, and embedding PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    collection_name = f"pdf_{uuid.uuid4().hex[:8]}"
                    vectorstore = ingest_pdf(tmp_path, collection_name=collection_name)
                    st.session_state["rag_chain"] = build_rag_chain(vectorstore)
                    st.session_state["indexed_file_id"] = file_id
                    st.session_state["current_collection"] = collection_name
                    st.session_state["messages"] = []
                    st.success(f"Indexed: {uploaded_file.name}")
                finally:
                    os.remove(tmp_path)

# Main Navigation Tabs
tab_chat, tab_db = st.tabs(["💬 Chat Q&A", "📊 ChromaDB Explorer"])

with tab_chat:
    if "rag_chain" not in st.session_state:
        st.info("Upload a PDF in the sidebar to start asking questions.")
    else:
        # Display existing message history
        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Voice input expander
        with st.expander("🎙️ Ask via Voice", expanded=False):
            st.caption("Record your question using your microphone. Gemini will transcribe and search your document.")
            voice_recording = st.audio_input("Record a voice question", key="voice_recorder")
            if voice_recording is not None:
                audio_bytes = voice_recording.getvalue()
                audio_hash = f"audio_{hash(audio_bytes)}"
                if st.session_state.get("last_processed_audio") != audio_hash:
                    st.session_state["last_processed_audio"] = audio_hash
                    with st.spinner("Transcribing your voice question with Gemini..."):
                        try:
                            transcribed_text = transcribe_audio(
                                audio_bytes,
                                mime_type=voice_recording.type or "audio/wav"
                            )
                            if transcribed_text and transcribed_text.lower() != "no speech detected.":
                                st.session_state["pending_voice_question"] = transcribed_text
                                st.rerun()
                            else:
                                st.warning("No speech detected in audio. Please try speaking again.")
                        except Exception as e:
                            st.error(f"Error transcribing audio: {e}")

        # Check for user question from either voice or text
        user_question = None
        is_voice = False
        if "pending_voice_question" in st.session_state:
            user_question = st.session_state.pop("pending_voice_question")
            is_voice = True
        elif text_question := st.chat_input("Ask a question about your PDF (or use voice above)..."):
            user_question = text_question

        if user_question:
            display_text = f"🎙️ {user_question}" if is_voice else user_question
            st.session_state["messages"].append({"role": "user", "content": display_text})
            with st.chat_message("user"):
                st.markdown(display_text)

            with st.chat_message("assistant"):
                with st.spinner("Searching document and generating answer..."):
                    result = st.session_state["rag_chain"].invoke(user_question)
                    answer = result["answer"]
                    st.markdown(answer)

                    with st.expander("View retrieved source chunks"):
                        for i, doc in enumerate(result["docs"], start=1):
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Chunk {i} (Page {page})**")
                            st.caption(doc.page_content)

            st.session_state["messages"].append({"role": "assistant", "content": answer})

with tab_db:
    st.subheader("ChromaDB Vector Store Explorer")
    if not os.path.exists(PERSIST_DIR):
        st.info(f"Vector database not found at `{PERSIST_DIR}`. Ingest a document first.")
    else:
        client = get_chroma_client()
        all_cols = client.list_collections()

        if not all_cols:
            st.info("No collections found in ChromaDB yet.")
        else:
            col_names = [c.name for c in all_cols]
            default_idx = 0
            curr_col = st.session_state.get("current_collection")
            if curr_col and curr_col in col_names:
                default_idx = col_names.index(curr_col)

            selected_col_name = st.selectbox("Select Collection to Visualize", col_names, index=default_idx)
            target_col = client.get_collection(selected_col_name)
            col_data = target_col.get(include=["documents", "metadatas", "embeddings"])

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
                st.metric("Active Collection", selected_col_name)

            subtab_docs, subtab_pca = st.tabs(["📄 Chunks & Metadata", "🌌 2D Embedding Projection (PCA)"])

            with subtab_docs:
                if ids:
                    table_rows = []
                    for i, chunk_id in enumerate(ids):
                        meta = metas[i] if metas and i < len(metas) else {}
                        doc = docs[i] if docs and i < len(docs) else ""
                        table_rows.append({
                            "Chunk ID": chunk_id,
                            "Page": meta.get("page", 0) + 1 if isinstance(meta, dict) else "N/A",
                            "Characters": len(doc),
                            "Content Preview": doc[:120] + "..." if len(doc) > 120 else doc,
                            "Full Content": doc
                        })

                    df = pd.DataFrame(table_rows)
                    kw = st.text_input("🔍 Filter chunks by keyword", "")
                    filtered = df[df["Full Content"].str.contains(kw, case=False, na=False)] if kw else df

                    st.dataframe(filtered[["Chunk ID", "Page", "Characters", "Content Preview"]], use_container_width=True, hide_index=True)

                    st.markdown("**Inspect Chunk Content**")
                    selected_id = st.selectbox("Select Chunk ID:", ids, key="chunk_inspector")
                    if selected_id:
                        idx = ids.index(selected_id)
                        st.caption(f"Metadata: {metas[idx] if metas and idx < len(metas) else {}}")
                        st.text_area("Chunk Content", docs[idx], height=180, key="chunk_text_area")

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
                        "Page": [f"Page {m.get('page', 0) + 1}" if isinstance(m, dict) else "N/A" for m in metas]
                    })
                    st.scatter_chart(plot_df, x="x", y="y", color="Page", use_container_width=True)
                    st.caption("2D projection of embedding vectors via PCA. Closer clusters indicate semantic closeness.")
                else:
                    st.info("At least 2 embeddings are required for 2D semantic projection.")
