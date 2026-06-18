"""
Pydantic request/response models for the API.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


class IngestURLRequest(BaseModel):
    url: HttpUrl = Field(..., description="Website URL to scrape and ingest")
    use_playwright: bool = Field(
        default=False, description="Use Playwright for JS-heavy sites instead of plain requests"
    )
    skip_duplicates: bool = Field(default=True, description="Skip chunks that already exist (by content hash)")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Extra metadata to attach to chunks")


class RecrawlURLRequest(BaseModel):
    url: HttpUrl
    use_playwright: bool = False


class IngestResponse(BaseModel):
    success: bool
    summary: dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's query, may contain multiple questions")
    conversation_id: str = Field(
        default="default", description="ID used to maintain conversation memory across turns"
    )
    retrieval_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve per sub-query")


class SourceCitation(BaseModel):
    source: str
    title: str = ""


class SubQueryResult(BaseModel):
    question: str
    num_chunks_retrieved: int


class ChatResponse(BaseModel):
    answer: str
    sub_queries: list[SubQueryResult]
    sources: list[SourceCitation]
    conversation_id: str


class HealthResponse(BaseModel):
    status: str
    vector_store_provider: str
