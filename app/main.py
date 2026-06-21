"""
FastAPI application entry point for the RAG project.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_ingest import router as ingest_router
from app.api.schemas import HealthResponse
from app.core.config import get_settings
from app.core.logging import setup_logging

setup_logging()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="RAG Project API",
    description="Retrieval-Augmented Generation API built with FastAPI, LangGraph, and LangChain.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", vector_store_provider=settings.vector_store_provider)


@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    """Minimal HTML UI for manual testing of ingestion and chat endpoints."""
    return FileResponse(STATIC_DIR/ "index.html")
