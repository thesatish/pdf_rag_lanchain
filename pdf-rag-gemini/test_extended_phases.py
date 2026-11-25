import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import rag_pipeline


class TestExtendedRAGPhases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_embeddings = FakeEmbeddings(size=768)
        cls.sample_pdf_path = os.path.join(PROJECT_DIR, "sample.pdf")

    # =========================================================================
    # PHASE 1: Separation of Ingestion, Retrieval, Generation & DB Reuse
    # =========================================================================
    def test_phase1_database_reuse_without_reingestion(self):
        """Verify vectorstore can be reused from existing collection without re-ingesting."""
        # Create initial collection with fake embeddings
        col_name = "test_phase1_reuse"
        vstore1 = Chroma.from_documents(
            documents=[Document(page_content="Existing chunk in database", metadata={"doc_id": "doc_1"})],
            embedding=self.fake_embeddings,
            collection_name=col_name,
            persist_directory=rag_pipeline.PERSIST_DIR,
            client=rag_pipeline.get_chroma_client(),
        )

        # Reopen collection without ingesting
        vstore2 = rag_pipeline.get_or_create_vectorstore(
            collection_name=col_name,
            client=rag_pipeline.get_chroma_client(),
            embeddings=self.fake_embeddings,
        )
        self.assertIsInstance(vstore2, Chroma)
        results = vstore2.similarity_search("Existing chunk", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Existing chunk in database", results[0].page_content)

    def test_phase1_separation_of_concerns(self):
        """Verify ingestion, retrieval, and generation are decoupled modules."""
        docs = [
            Document(page_content="Separated retrieval context.", metadata={"page": 0, "filename": "test.pdf"})
        ]
        # Ingestion formatting check
        formatted = rag_pipeline.format_docs(docs)
        self.assertIn("[Document: test.pdf | Page 1]", formatted)

        # Generation chain creation
        mock_llm = FakeListChatModel(responses=["Decoupled generation output."])
        gen_chain = rag_pipeline.build_generation_chain(llm=mock_llm)
        out = gen_chain.invoke({"docs": docs, "question": "Test question"})
        self.assertEqual(out, "Decoupled generation output.")

    # =========================================================================
    # PHASE 2: Document Management (Multiple PDFs, Duplicates, Delete, Update)
    # =========================================================================
    def test_phase2_compute_file_hash(self):
        hash1 = rag_pipeline.compute_file_hash(self.sample_pdf_path)
        self.assertEqual(len(hash1), 64)
        hash_bytes = rag_pipeline.compute_file_hash(b"dummy pdf bytes")
        self.assertEqual(len(hash_bytes), 64)

    @patch("rag_pipeline.get_embeddings")
    def test_phase2_duplicate_prevention(self, mock_emb):
        import uuid
        mock_emb.return_value = self.fake_embeddings
        col_name = f"test_phase2_dup_{uuid.uuid4().hex[:8]}"

        # First ingestion
        res1 = rag_pipeline.ingest_pdf(
            self.sample_pdf_path,
            collection_name=col_name,
            prevent_duplicates=True,
            return_summary=True,
        )
        self.assertEqual(res1["status"], "success")
        self.assertGreater(res1["chunks_added"], 0)

        # Second ingestion with same file should be detected as duplicate
        res2 = rag_pipeline.ingest_pdf(
            self.sample_pdf_path,
            collection_name=col_name,
            prevent_duplicates=True,
            return_summary=True,
        )
        self.assertEqual(res2["status"], "duplicate")
        self.assertEqual(res2["chunks_added"], 0)
        self.assertIn("Duplicate document detected", res2["message"])

    @patch("rag_pipeline.get_embeddings")
    def test_phase2_document_listing_and_deletion(self, mock_emb):
        import uuid
        mock_emb.return_value = self.fake_embeddings
        col_name = f"test_phase2_del_{uuid.uuid4().hex[:8]}"

        # Ingest document
        ingest_res = rag_pipeline.ingest_pdf(
            self.sample_pdf_path,
            collection_name=col_name,
            prevent_duplicates=False,
            filename="doc_to_delete.pdf",
            return_summary=True,
        )
        self.assertEqual(ingest_res["status"], "success")

        # List documents in collection
        doc_list = rag_pipeline.list_documents_in_collection(col_name)
        self.assertTrue(any(d["filename"] == "doc_to_delete.pdf" for d in doc_list))

        # Delete document
        del_res = rag_pipeline.delete_document(col_name, filename="doc_to_delete.pdf")
        self.assertEqual(del_res["status"], "success")
        self.assertGreater(del_res["deleted_count"], 0)

        # Verify it no longer appears
        updated_list = rag_pipeline.list_documents_in_collection(col_name)
        self.assertFalse(any(d["filename"] == "doc_to_delete.pdf" for d in updated_list))

    # =========================================================================
    # PHASE 3: Advanced Retrieval (MMR, Thresholds, Metadata Filters, Debugger)
    # =========================================================================
    def test_phase3_retriever_modes_and_metadata_filtering(self):
        col_name = "test_phase3_retrieval"
        docs = [
            Document(page_content="Doc Alpha topic machine learning", metadata={"filename": "alpha.pdf", "page": 0, "chunk_id": "c1"}),
            Document(page_content="Doc Beta topic deep learning neural networks", metadata={"filename": "beta.pdf", "page": 0, "chunk_id": "c2"}),
        ]
        vstore = Chroma.from_documents(
            documents=docs,
            embedding=self.fake_embeddings,
            collection_name=col_name,
            persist_directory=rag_pipeline.PERSIST_DIR,
            client=rag_pipeline.get_chroma_client(),
        )

        # 1. Similarity Retriever
        sim_retriever = rag_pipeline.get_retriever(vstore, search_type="similarity", k=1)
        self.assertEqual(sim_retriever.search_type, "similarity")

        # 2. MMR Retriever
        mmr_retriever = rag_pipeline.get_retriever(vstore, search_type="mmr", k=2, fetch_k=10, lambda_mult=0.7)
        self.assertEqual(mmr_retriever.search_type, "mmr")
        self.assertEqual(mmr_retriever.search_kwargs["lambda_mult"], 0.7)

        # 3. Metadata Filtering: only retrieve from beta.pdf
        filtered_retriever = rag_pipeline.get_retriever(
            vstore,
            search_type="similarity",
            k=2,
            metadata_filter={"filename": "beta.pdf"},
        )
        res = filtered_retriever.invoke("machine learning")
        for d in res:
            self.assertEqual(d.metadata["filename"], "beta.pdf")

    def test_phase3_retrieval_debugging(self):
        col_name = "test_phase3_debug"
        docs = [
            Document(page_content="Document content for latency and score testing", metadata={"filename": "test.pdf", "page": 0, "chunk_id": "debug_c1"}),
        ]
        vstore = Chroma.from_documents(
            documents=docs,
            embedding=self.fake_embeddings,
            collection_name=col_name,
            persist_directory=rag_pipeline.PERSIST_DIR,
            client=rag_pipeline.get_chroma_client(),
        )

        result = rag_pipeline.retrieve_with_debug(vstore, query="latency testing", k=1)
        self.assertIn("docs", result)
        self.assertIn("debug_info", result)
        debug = result["debug_info"]
        self.assertIn("latency_ms", debug)
        self.assertGreaterEqual(debug["latency_ms"], 0.0)
        self.assertEqual(debug["num_retrieved"], 1)
        self.assertEqual(len(debug["items"]), 1)
        item = debug["items"][0]
        self.assertEqual(item["chunk_id"], "debug_c1")
        self.assertIn("score", item)

    # =========================================================================
    # PHASE 4: RAG Evaluation (Metrics, Benchmarking, Synthetic QA)
    # =========================================================================
    def test_phase4_token_f1_metrics(self):
        scores = rag_pipeline.token_f1_score("the quick brown fox", "the fast brown fox")
        self.assertGreater(scores["f1"], 0.6)
        self.assertAlmostEqual(scores["recall"], 0.75, places=2)

    def test_phase4_recall_at_k(self):
        docs = [
            Document(page_content="text 1", metadata={"chunk_id": "target_123", "page": 2, "filename": "sample.pdf"}),
            Document(page_content="text 2", metadata={"chunk_id": "other_456", "page": 3, "filename": "sample.pdf"}),
        ]
        # Match by chunk ID
        self.assertEqual(rag_pipeline.recall_at_k(docs, target_chunk_id="target_123"), 1.0)
        self.assertEqual(rag_pipeline.recall_at_k(docs, target_chunk_id="not_present"), 0.0)
        # Match by page number
        self.assertEqual(rag_pipeline.recall_at_k(docs, target_page=3, target_filename="sample.pdf"), 1.0)

    def test_phase4_synthetic_qa_generation(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"question": "What is the primary feature of ChromaDB?", "ground_truth_answer": "Vector similarity search."}'
        )
        sample_doc = Document(page_content="ChromaDB provides fast vector similarity search.", metadata={"chunk_id": "c1", "page": 0})
        qa_data = rag_pipeline.generate_synthetic_qa([sample_doc], num_questions=1, llm=mock_llm)
        self.assertEqual(len(qa_data), 1)
        self.assertEqual(qa_data[0]["question"], "What is the primary feature of ChromaDB?")
        self.assertEqual(qa_data[0]["ground_truth_answer"], "Vector similarity search.")

    def test_phase4_evaluate_rag_pipeline(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="0.95")  # mock judge score

        dataset = [
            {
                "id": "q1",
                "question": "What is Python?",
                "ground_truth_answer": "A programming language.",
                "source_chunk_id": "c_py",
            }
        ]

        def dummy_invoke(query):
            return {
                "answer": "Python is a popular programming language.",
                "docs": [Document(page_content="Python is a programming language.", metadata={"chunk_id": "c_py"})],
            }

        eval_res = rag_pipeline.evaluate_rag_pipeline(dummy_invoke, dataset, llm=mock_llm)
        self.assertIn("summary", eval_res)
        summary = eval_res["summary"]
        self.assertEqual(summary["mean_recall_at_k"], 1.0)
        self.assertEqual(summary["total_evaluated"], 1)

    # =========================================================================
    # PHASE 5: Advanced RAG Techniques (Query Rewriting, Hybrid BM25, Reranking, Compression)
    # =========================================================================
    def test_phase5_query_rewriting(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="quantum computing applications\nquantum algorithms overview\nquantum hardware state"
        )
        queries = rag_pipeline.rewrite_query("quantum computing", mode="expand", llm=mock_llm)
        self.assertEqual(len(queries), 3)
        self.assertIn("quantum computing applications", queries)

    def test_phase5_pure_python_bm25_search(self):
        bm25 = rag_pipeline.BM25Index()
        corpus = [
            Document(page_content="Artificial intelligence and machine learning models in healthcare.", metadata={"id": 1}),
            Document(page_content="The cooking recipe for traditional Italian sourdough pizza.", metadata={"id": 2}),
            Document(page_content="Deep neural networks and artificial intelligence algorithms.", metadata={"id": 3}),
        ]
        bm25.index_documents(corpus)
        results = bm25.search("artificial intelligence healthcare", top_k=2)
        self.assertEqual(len(results), 2)
        # First result should be document 1 due to exact terms 'healthcare' and 'artificial intelligence'
        self.assertIn("healthcare", results[0][0].page_content)

    def test_phase5_reciprocal_rank_fusion(self):
        docA = Document(page_content="Doc A", metadata={"chunk_id": "A"})
        docB = Document(page_content="Doc B", metadata={"chunk_id": "B"})
        docC = Document(page_content="Doc C", metadata={"chunk_id": "C"})

        dense_list = [docA, docB]
        sparse_list = [docB, docC]

        fused = rag_pipeline.reciprocal_rank_fusion([dense_list, sparse_list], k=60, top_n=3)
        self.assertEqual(len(fused), 3)
        # Doc B appeared in both lists, so its RRF score is highest
        self.assertEqual(fused[0][0].metadata["chunk_id"], "B")

    def test_phase5_contextual_compression(self):
        chunk_content = (
            "Python was created by Guido van Rossum. It was first released in 1991. "
            "Python emphasizes code readability with significant indentation. "
            "The weather in Antarctica is extremely cold and windy."
        )
        doc = Document(page_content=chunk_content, metadata={"chunk_id": "c_py"})
        compressed = rag_pipeline.compress_context("What is Python syntax and indentation?", [doc], max_sentences=2)
        self.assertEqual(len(compressed), 1)
        self.assertIn("indentation", compressed[0].page_content)
        self.assertNotIn("Antarctica", compressed[0].page_content)


if __name__ == "__main__":
    unittest.main()
