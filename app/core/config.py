"""
Application configuration using pydantic-settings.
Loads from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: SecretStr = SecretStr("")
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Vector store
    vector_store_provider: str = "pinecone"  # "pinecone" | "chroma"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "rag-project-index"
    pinecone_cloud: str = "aws"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Conversation
    max_conversation_history: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
