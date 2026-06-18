"""
Text chunking utilities shared by both ingestion flows (website + file upload).
"""
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def compute_content_hash(text: str) -> str:
    """Stable hash of normalized content, used for duplicate detection."""
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def chunk_text(text: str, metadata: dict) -> list[Document]:
    """Split raw text into Document chunks, attaching metadata and a content hash."""
    splitter = get_text_splitter()
    chunks = splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = dict(metadata)
        chunk_metadata["chunk_index"] = i
        chunk_metadata["content_hash"] = compute_content_hash(chunk)
        documents.append(Document(page_content=chunk, metadata=chunk_metadata))
    return documents
