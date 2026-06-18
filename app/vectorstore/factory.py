"""
Factory for creating a vector store instance based on configuration.

This is the single point where the rest of the application decides which
vector database backend to use. Switching backends only requires changing
the VECTOR_STORE_PROVIDER setting (and adding a new adapter if needed).
"""
from functools import lru_cache

from app.core.config import get_settings
from app.vectorstore.base import BaseVectorStore


@lru_cache
def get_vector_store() -> BaseVectorStore:
    settings = get_settings()
    provider = settings.vector_store_provider.lower()

    if provider == "pinecone":
        from app.vectorstore.pinecone_store import PineconeVectorStore

        return PineconeVectorStore()
    elif provider == "chroma":
        from app.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore()
    else:
        raise ValueError(f"Unsupported vector store provider: {provider}")
