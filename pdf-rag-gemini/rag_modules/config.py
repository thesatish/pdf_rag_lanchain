import os
import warnings
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_persist = os.getenv("PERSIST_DIR", "./chroma_db")
PERSIST_DIR = os.path.abspath(os.path.join(BASE_DIR, raw_persist)) if not os.path.isabs(raw_persist) else raw_persist

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")

_chroma_client = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a shared singleton ChromaDB persistent client to prevent SQLite lock collisions."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    return _chroma_client


def reset_chroma_client():
    """Reset the singleton ChromaDB client instance (useful in tests or db rebuilds)."""
    global _chroma_client
    _chroma_client = None


def get_embeddings(model_name: str = None) -> GoogleGenerativeAIEmbeddings:
    """Initialize Gemini embedding model (768 dims balances speed and accuracy)."""
    model = model_name or EMBEDDING_MODEL
    return GoogleGenerativeAIEmbeddings(
        model=model,
        output_dimensionality=768,
    )


def get_llm(model_name: str = None, temperature: float = 0.2, max_retries: int = 2) -> ChatGoogleGenerativeAI:
    """Initialize Gemini Chat LLM."""
    model = model_name or LLM_MODEL
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

