import os
import sys
import unittest
import uuid
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from server import app
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import rag_pipeline


class TestFastAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_pdf_path = os.path.join(PROJECT_DIR, "sample.pdf")
        cls.test_col = f"test_api_{uuid.uuid4().hex[:8]}"

    def test_healthcheck(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("llm_model", data)
        self.assertIn("embedding_model", data)

    def test_collections_lifecycle(self):
        col_name = f"col_crud_{uuid.uuid4().hex[:6]}"
        # Create
        res_post = self.client.post("/api/collections", json={"name": col_name})
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json()["name"], col_name)

        # List
        res_get = self.client.get("/api/collections")
        self.assertEqual(res_get.status_code, 200)
        cols = [c["name"] for c in res_get.json()]
        self.assertIn(col_name, cols)

        # Delete
        res_del = self.client.delete(f"/api/collections/{col_name}")
        self.assertEqual(res_del.status_code, 200)
        self.assertEqual(res_del.json()["status"], "success")

    @patch("rag_pipeline.get_embeddings")
    def test_upload_and_documents(self, mock_emb):
        mock_emb.return_value = FakeEmbeddings(size=768)

        # Upload sample PDF
        with open(self.sample_pdf_path, "rb") as f:
            files = [("files", ("test_upload.pdf", f.read(), "application/pdf"))]
            resp = self.client.post(
                "/api/documents/upload",
                files=files,
                data={"collection_name": self.test_col, "prevent_duplicates": "false"},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_files"], 1)
        self.assertEqual(data["results"][0]["status"], "success")

        # Get documents
        doc_resp = self.client.get(f"/api/documents?collection_name={self.test_col}")
        self.assertEqual(doc_resp.status_code, 200)
        docs = doc_resp.json()
        self.assertTrue(any(d["filename"] == "test_upload.pdf" for d in docs))

        # Explorer
        exp_resp = self.client.get(f"/api/explorer?collection_name={self.test_col}")
        self.assertEqual(exp_resp.status_code, 200)
        exp_data = exp_resp.json()
        self.assertGreater(exp_data["total_chunks"], 0)

        # Delete document
        del_resp = self.client.delete(f"/api/documents/test_upload.pdf?collection_name={self.test_col}")
        self.assertEqual(del_resp.status_code, 200)

    @patch("rag_pipeline.get_embeddings")
    def test_retrieval_debug(self, mock_emb):
        mock_emb.return_value = FakeEmbeddings(size=768)
        # Populate test document
        col_name = f"debug_col_{uuid.uuid4().hex[:6]}"
        vstore = Chroma.from_documents(
            documents=[Document(page_content="FastAPI is a modern web framework", metadata={"filename": "api.pdf", "page": 0, "chunk_id": "c1"})],
            embedding=mock_emb.return_value,
            collection_name=col_name,
            persist_directory=rag_pipeline.PERSIST_DIR,
            client=rag_pipeline.get_chroma_client(),
        )

        resp = self.client.post(
            "/api/retrieve/debug",
            json={
                "collection_name": col_name,
                "query": "web framework",
                "search_type": "similarity",
                "k": 1,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("latency_ms", data)
        self.assertEqual(data["num_retrieved"], 1)
        self.assertEqual(len(data["items"]), 1)
        self.assertEqual(data["items"][0]["filename"], "api.pdf")

    @patch("server.build_rag_chain")
    def test_chat_handles_503_gracefully(self, mock_build_chain):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("503 UNAVAILABLE: Model experiencing high demand")
        mock_build_chain.return_value = mock_chain

        resp = self.client.post(
            "/api/chat",
            json={
                "collection_name": self.test_col,
                "query": "Test query when model is down",
                "mode": "standard",
            },
        )
        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertIn("detail", data)
        self.assertIn("temporarily unavailable or experiencing high demand", data["detail"])

    @patch("server.build_rag_chain")
    def test_chat_handles_429_quota_exceeded(self, mock_build_chain):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit quota exceeded")
        mock_build_chain.return_value = mock_chain

        resp = self.client.post(
            "/api/chat",
            json={
                "collection_name": self.test_col,
                "query": "Test query when quota exceeded",
                "mode": "standard",
            },
        )
        self.assertEqual(resp.status_code, 429)
        self.assertIn("quota or rate limit exceeded", resp.json()["detail"])

    @patch("server.build_rag_chain")
    def test_chat_handles_401_invalid_key(self, mock_build_chain):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("401 UNAUTHENTICATED: Invalid API_KEY")
        mock_build_chain.return_value = mock_chain

        resp = self.client.post(
            "/api/chat",
            json={
                "collection_name": self.test_col,
                "query": "Test query when key invalid",
                "mode": "standard",
            },
        )
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid or missing Gemini API key", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()

