import os
import warnings

# Suppress Python 3.9 deprecation warnings from google-auth/google-api-core
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize Gemini embedding model (768 dims balances speed and accuracy)."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=768,
    )


def ingest_pdf(pdf_path: str, collection_name: str = "pdf_rag") -> Chroma:
    """Load a PDF, split into chunks, embed, and store in ChromaDB."""
    # 1. Load PDF pages
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split pages into overlapping chunks to preserve context across boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed chunks and index them into ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


def format_docs(docs) -> str:
    """Combine retrieved chunks with their page numbers for the prompt."""
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma):
    """Build an LCEL RAG chain that returns both the answer and source chunks."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.2,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise document assistant. Answer the user's question "
                "using ONLY the provided context. Cite page numbers (e.g., [Page 3]) "
                "when referencing facts. If the answer is not in the context, say "
                "'I cannot find that information in the uploaded PDF.'\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    # Retrieve context once, pass it to the LLM, and keep source docs for display
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = RunnableParallel(
        {"docs": retriever, "question": RunnablePassthrough()}
    ).assign(answer=answer_chain)

    return rag_chain
