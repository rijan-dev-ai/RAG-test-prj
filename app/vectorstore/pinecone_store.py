"""
Pinecone implementation of the BaseVectorStore interface.
"""
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings
from app.core.logging import get_logger
from app.vectorstore.base import BaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)


class PineconeVectorStore(BaseVectorStore):
    """Vector store backed by Pinecone serverless indexes."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings

        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

        self._client = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._ensure_index()
        self._index = self._client.Index(self._index_name)

    def _ensure_index(self) -> None:
        """Create the Pinecone index if it doesn't already exist."""
        existing = [idx["name"] for idx in self._client.list_indexes()]
        if self._index_name not in existing:
            # Determine embedding dimension dynamically based on model
            dimension = self._embedding_dimension()
            logger.info("Creating Pinecone index '%s' (dim=%d)", self._index_name, dimension)
            self._client.create_index(
                name=self._index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_environment,
                ),
            )

    def _embedding_dimension(self) -> int:
        # text-embedding-3-small -> 1536, text-embedding-3-large -> 3072
        model = self._settings.openai_embedding_model
        if "large" in model:
            return 3072
        return 1536

    def add_documents(self, documents: list[Document], ids: Optional[list[str]] = None) -> list[str]:
        if not documents:
            return []

        if ids is None:
            ids = [f"doc-{i}-{hash(doc.page_content) & 0xFFFFFFFF}" for i, doc in enumerate(documents)]

        texts = [doc.page_content for doc in documents]
        vectors = self._embeddings.embed_documents(texts)

        upsert_payload = []
        for doc_id, vector, doc in zip(ids, vectors, documents):
            metadata = dict(doc.metadata)
            metadata["text"] = doc.page_content[:4000]  # Pinecone metadata size limits
            upsert_payload.append({"id": doc_id, "values": vector, "metadata": metadata})

        # Pinecone recommends batching upserts (<= 100 records per call)
        batch_size = 100
        for i in range(0, len(upsert_payload), batch_size):
            batch = upsert_payload[i : i + batch_size]
            self._index.upsert(vectors=batch)

        logger.info("Upserted %d vectors into Pinecone index '%s'", len(upsert_payload), self._index_name)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        query_vector = self._embeddings.embed_query(query)
        results = self._index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            filter=filter,
        )

        documents = []
        for match in results.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            text = metadata.pop("text", "")
            metadata["score"] = match.get("score")
            metadata["id"] = match.get("id")
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def delete(self, ids: Optional[list[str]] = None, filter: Optional[dict[str, Any]] = None) -> None:
        if ids:
            self._index.delete(ids=ids)
        elif filter:
            self._index.delete(filter=filter)
        else:
            logger.warning("delete() called without ids or filter; no-op")

    def document_exists(self, content_hash: str) -> bool:
        """Query Pinecone for any vector with this content_hash in metadata."""
        try:
            dummy_vector = [0.0] * self._embedding_dimension()
            results = self._index.query(
                vector=dummy_vector,
                top_k=1,
                include_metadata=False,
                filter={"content_hash": {"$eq": content_hash}},
            )
            return len(results.get("matches", [])) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("document_exists check failed: %s", exc)
            return False
