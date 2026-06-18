"""
Abstract base class for vector stores.

Any vector database backend (Pinecone, Chroma, Weaviate, etc.) must
implement this interface so the rest of the application is decoupled
from the specific vector DB implementation. This makes it easy to swap
vector stores in the future by writing a new adapter class.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Common interface every vector store adapter must implement."""

    @abstractmethod
    def add_documents(self, documents: list[Document], ids: Optional[list[str]] = None) -> list[str]:
        """Embed and store documents. Returns the list of stored IDs."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """Return the top-k most similar documents to the query."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: Optional[list[str]] = None, filter: Optional[dict[str, Any]] = None) -> None:
        """Delete documents by ID or by metadata filter."""
        raise NotImplementedError

    @abstractmethod
    def document_exists(self, content_hash: str) -> bool:
        """Check whether a document chunk with this content hash already exists.

        Used for duplicate content detection.
        """
        raise NotImplementedError
