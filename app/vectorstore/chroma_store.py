"""
Chroma implementation of BaseVectorStore.

Used as a local, no-API-key-required fallback for development and testing,
and to demonstrate that the vector store layer is swappable.
"""
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings
from app.core.logging import get_logger
from app.vectorstore.base import BaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by a local Chroma persistent collection."""

    def __init__(self, persist_directory: str = "./data/chroma") -> None:
        settings = get_settings()
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self._store = Chroma(
            collection_name="rag-project",
            embedding_function=self._embeddings,
            persist_directory=persist_directory,
        )

    def add_documents(self, documents: list[Document], ids: Optional[list[str]] = None) -> list[str]:
        if not documents:
            return []
        return self._store.add_documents(documents=documents, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        results = self._store.similarity_search_with_score(query, k=k, filter=filter)
        documents = []
        for doc, score in results:
            doc.metadata["score"] = score
            documents.append(doc)
        return documents

    def delete(self, ids: Optional[list[str]] = None, filter: Optional[dict[str, Any]] = None) -> None:
        if ids:
            self._store.delete(ids=ids)
        elif filter:
            # Chroma delete by where-filter
            self._store.delete(where=filter)
        else:
            logger.warning("delete() called without ids or filter; no-op")

    def document_exists(self, content_hash: str) -> bool:
        results = self._store.get(where={"content_hash": content_hash}, limit=1)
        return len(results.get("ids", [])) > 0
