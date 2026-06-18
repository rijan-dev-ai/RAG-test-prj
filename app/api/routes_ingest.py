"""
API routes for data ingestion: website URLs and file uploads.
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import IngestResponse, IngestURLRequest, RecrawlURLRequest
from app.core.logging import get_logger
from app.ingestion.file_processor import SUPPORTED_EXTENSIONS, ingest_file
from app.ingestion.web_scraper import ingest_website, recrawl_website
from app.vectorstore.factory import get_vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/url", response_model=IngestResponse)
def ingest_url(request: IngestURLRequest) -> IngestResponse:
    """Ingest content from a website URL."""
    vector_store = get_vector_store()
    try:
        summary = ingest_website(
            url=str(request.url),
            vector_store=vector_store,
            use_playwright=request.use_playwright,
            skip_duplicates=request.skip_duplicates,
            extra_metadata=request.metadata,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to ingest URL %s", request.url)
        raise HTTPException(status_code=400, detail=f"Failed to ingest URL: {exc}") from exc

    return IngestResponse(success=True, summary=summary)


@router.post("/url/recrawl", response_model=IngestResponse)
def recrawl_url(request: RecrawlURLRequest) -> IngestResponse:
    """Re-crawl an already-ingested website: remove old chunks and re-ingest fresh content."""
    vector_store = get_vector_store()
    try:
        summary = recrawl_website(
            url=str(request.url),
            vector_store=vector_store,
            use_playwright=request.use_playwright,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to re-crawl URL %s", request.url)
        raise HTTPException(status_code=400, detail=f"Failed to re-crawl URL: {exc}") from exc

    return IngestResponse(success=True, summary=summary)


@router.post("/file", response_model=IngestResponse)
async def ingest_uploaded_file(file: UploadFile = File(...)) -> IngestResponse:
    """Ingest content from an uploaded file (PDF, DOCX, TXT, MD)."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    vector_store = get_vector_store()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        summary = ingest_file(
            file_path=tmp_path,
            vector_store=vector_store,
            original_filename=file.filename,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to ingest file %s", file.filename)
        raise HTTPException(status_code=400, detail=f"Failed to ingest file: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(success=True, summary=summary)
