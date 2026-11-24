import os
import base64
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import get_llm, LLM_MODEL


def format_docs(docs: List[Document]) -> str:
    """Combine retrieved chunks with document name and page numbers for prompt context."""
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        filename = doc.metadata.get("filename", "")
        header = f"[Document: {filename} | Page {page}]" if filename else f"[Page {page}]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_qa_prompt(custom_system_prompt: Optional[str] = None) -> ChatPromptTemplate:
    """Build the ChatPromptTemplate for question answering with context."""
    system_prompt = custom_system_prompt or (
        "You are a precise document assistant. Answer the user's question "
        "using ONLY the provided context. Cite source files and page numbers "
        "(e.g., [Document: manual.pdf | Page 3] or [Page 3]) when referencing facts. "
        "If the answer is not in the context, say "
        "'I cannot find that information in the uploaded PDF.'\n\n"
        "Context:\n{context}"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )


def build_generation_chain(llm: Optional[ChatGoogleGenerativeAI] = None, prompt: Optional[ChatPromptTemplate] = None):
    """Build an LCEL generation chain that maps context and question to a formatted answer."""
    if llm is None:
        llm = get_llm()
    if prompt is None:
        prompt = get_qa_prompt()

    chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def generate_answer(llm: ChatGoogleGenerativeAI, docs: List[Document], question: str) -> str:
    """Directly generate an answer from a list of documents and question."""
    chain = build_generation_chain(llm=llm)
    return chain.invoke({"docs": docs, "question": question})


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav", llm: Optional[ChatGoogleGenerativeAI] = None) -> str:
    """Transcribe speech audio into text using Gemini's native multimodal capabilities."""
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    if llm is None:
        llm = get_llm(temperature=0.0)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Transcribe the spoken question or statement in this audio accurately. "
                    "Output ONLY the verbatim transcription, with no quotation marks, prefixes, "
                    "or conversational explanations. If no speech is heard, return 'No speech detected.'"
                ),
            },
            {
                "type": "media",
                "mime_type": mime_type,
                "data": b64_audio,
            },
        ]
    )
    response = llm.invoke([message])
    return response.content.strip()

