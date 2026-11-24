import os
import sys
import unittest
import py_compile
import shutil

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from unittest.mock import patch, MagicMock

import rag_pipeline


class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_dir = os.path.join(PROJECT_DIR, "test_chroma_db")
        cls.sample_pdf_path = os.path.join(PROJECT_DIR, "sample.pdf")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db_dir):
            shutil.rmtree(cls.test_db_dir, ignore_errors=True)

    def test_format_docs(self):
        docs = [
            Document(page_content="First chunk content", metadata={"page": 0}),
            Document(page_content="Second chunk content", metadata={"page": 1}),
        ]
        formatted = rag_pipeline.format_docs(docs)
        self.assertIn("[Page 1]\nFirst chunk content", formatted)
        self.assertIn("[Page 2]\nSecond chunk content", formatted)
        self.assertIn("---", formatted)

    def test_get_embeddings_configuration(self):
        embeddings = rag_pipeline.get_embeddings()
        # GoogleGenerativeAIEmbeddings ensures 'models/' prefix
        self.assertTrue(
            embeddings.model == rag_pipeline.EMBEDDING_MODEL or 
            embeddings.model == f"models/{rag_pipeline.EMBEDDING_MODEL}"
        )
        self.assertIsInstance(embeddings, rag_pipeline.GoogleGenerativeAIEmbeddings)

    @patch("rag_pipeline.get_embeddings")
    def test_ingest_pdf_with_chroma(self, mock_get_embeddings):
        # Use FakeEmbeddings so no external Google API calls are needed for unit tests
        mock_get_embeddings.return_value = FakeEmbeddings(size=768)

        vectorstore = rag_pipeline.ingest_pdf(
            self.sample_pdf_path,
            collection_name="unit_test_pdf"
        )
        self.assertIsInstance(vectorstore, Chroma)

        # Verify similarity retrieval works on indexed documents
        results = vectorstore.similarity_search("ChromaDB embeddings", k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any(len(doc.page_content) > 0 for doc in results))

    @patch("rag_pipeline.ChatGoogleGenerativeAI")
    def test_build_rag_chain_execution(self, mock_chat_cls):
        # FakeChatModel returns pre-set answers and implements the LangChain Runnable interface
        mock_chat_cls.return_value = FakeListChatModel(
            responses=["LangChain and Gemini provide modern document retrieval capabilities."]
        )

        fake_embeddings = FakeEmbeddings(size=768)
        vectorstore = Chroma.from_documents(
            documents=[
                Document(
                    page_content="LangChain and Gemini provide modern document retrieval capabilities.",
                    metadata={"page": 0}
                )
            ],
            embedding=fake_embeddings,
            collection_name="chain_test_collection"
        )

        chain = rag_pipeline.build_rag_chain(vectorstore)
        result = chain.invoke("What does Gemini provide?")

        self.assertIn("answer", result)
        self.assertIn("docs", result)
        self.assertEqual(len(result["docs"]), 1)
        self.assertEqual(result["docs"][0].metadata["page"], 0)
        self.assertIn("LangChain and Gemini", result["answer"])

    @patch("rag_pipeline.ChatGoogleGenerativeAI")
    def test_transcribe_audio(self, mock_chat_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="What are the key security practices?")
        mock_chat_cls.return_value = mock_llm

        dummy_audio = b"RIFF\x00\x00\x00\x00WAVEfmt "
        transcript = rag_pipeline.transcribe_audio(dummy_audio, mime_type="audio/wav")

        self.assertEqual(transcript, "What are the key security practices?")
        mock_llm.invoke.assert_called_once()

    def test_streamlit_app_compilation(self):
        app_path = os.path.join(PROJECT_DIR, "app.py")
        compiled = py_compile.compile(app_path)
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main()
