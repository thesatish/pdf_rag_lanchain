import os
import streamlit as st
import chromadb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import get_chroma_client, PERSIST_DIR

st.set_page_config(page_title="ChromaDB Visualizer", layout="wide")
st.title("ChromaDB Vector Store Explorer")

if not os.path.exists(PERSIST_DIR):
    st.warning(f"Database directory not found at `{PERSIST_DIR}`. Ingest a PDF first.")
    st.stop()

# Initialize Chroma client
client = get_chroma_client()
collections = client.list_collections()

if not collections:
    st.info("No collections found in ChromaDB yet.")
    st.stop()

# Sidebar: collection selector
col_names = [c.name for c in collections]
selected_col_name = st.sidebar.selectbox("Select Collection", col_names)

col = client.get_collection(selected_col_name)
data = col.get(include=["documents", "metadatas", "embeddings"])

ids = data.get("ids", [])
docs = data.get("documents", [])
metas = data.get("metadatas", [])
embeddings = data.get("embeddings", None)

# Summary Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Chunks", len(ids))
with col2:
    dim = len(embeddings[0]) if (embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None) else "N/A"
    st.metric("Embedding Dimension", dim)
with col3:
    st.metric("Collection Name", selected_col_name)

st.divider()

# Tab 1: Chunk Data & Search, Tab 2: 2D Embedding Scatter Plot
tab_data, tab_plot = st.tabs(["📄 Chunk Documents & Metadata", "🌌 2D Embedding Space (PCA)"])

with tab_data:
    if ids:
        # Build DataFrame
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

        # Search bar
        search_kw = st.text_input("🔍 Search chunks by keyword", "")
        if search_kw:
            filtered_df = df[df["Full Content"].str.contains(search_kw, case=False, na=False)]
        else:
            filtered_df = df

        st.dataframe(
            filtered_df[["Chunk ID", "Page", "Characters", "Content Preview"]],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Inspect Chunk Details")
        selected_id = st.selectbox("Select a Chunk ID to view full content:", ids)
        if selected_id:
            idx = ids.index(selected_id)
            st.markdown(f"**Chunk ID:** `{selected_id}` | **Page:** `{df.iloc[idx]['Page']}`")
            st.text_area("Full Document Text", docs[idx], height=220)
            st.json(metas[idx] if metas and idx < len(metas) else {})

with tab_plot:
    if embeddings is not None and len(embeddings) > 1 and embeddings[0] is not None:
        st.markdown("Projecting high-dimensional embeddings down to 2D using **Principal Component Analysis (PCA)**:")
        embs_arr = np.array(embeddings)
        
        # Center embeddings
        embs_centered = embs_arr - np.mean(embs_arr, axis=0)
        u, s, _ = np.linalg.svd(embs_centered, full_matrices=False)
        coords_2d = u[:, :2] * s[:2]
        
        plot_df = pd.DataFrame({
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "Chunk ID": ids,
            "Page": [f"Page {m.get('page', 0) + 1}" if isinstance(m, dict) else "N/A" for m in metas]
        })

        st.scatter_chart(plot_df, x="x", y="y", color="Page", use_container_width=True)
        st.caption("Each point represents a text chunk. Chunks closer together have higher semantic similarity.")
    else:
        st.info("At least 2 embeddings are required to generate a 2D projection.")
