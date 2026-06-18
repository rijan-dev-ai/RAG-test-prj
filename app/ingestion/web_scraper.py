"""
Website ingestion: scrape a URL, clean text, chunk it, embed and store it.
"""
import hashlib
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from app.core.logging import get_logger
from app.ingestion.chunking import chunk_text, compute_content_hash
from app.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAGIngestBot/1.0; "
        "+https://example.com/bot-info)"
    )
}


def fetch_html(url: str, timeout: int = 20) -> str:
    """Fetch raw HTML for a URL using requests.

    For JavaScript-heavy sites, swap this out for a Playwright-based fetch
    (see fetch_html_with_playwright below).
    """
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_html_with_playwright(url: str, timeout: int = 50000) -> str:
    """Fetch fully-rendered HTML using Playwright (for JS-heavy sites).

    Requires `playwright install chromium` to have been run.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=timeout, wait_until="networkidle")
        html = page.content()
        browser.close()
    return html


def clean_html_to_text(html: str) -> tuple[str, str]:
    """Extract main readable text and title from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Prefer <main> or <article> if present, else fall back to <body>
    main = soup.find("main") or soup.find("article") or soup.body or soup

    text = main.get_text(separator="\n")
    # Collapse excessive whitespace/blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)
    return cleaned, title


def ingest_website(
    url: str,
    vector_store: BaseVectorStore,
    use_playwright: bool = False,
    skip_duplicates: bool = True,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """Full pipeline: scrape -> clean -> chunk -> embed -> store.

    Returns a summary dict with counts of chunks added/skipped.
    """
    logger.info("Ingesting website: %s (playwright=%s)", url, use_playwright)

    html = fetch_html_with_playwright(url) if use_playwright else fetch_html(url)
    text, title = clean_html_to_text(html)

    if not text:
        logger.warning("No text extracted from %s", url)
        return {"url": url, "chunks_added": 0, "chunks_skipped": 0, "title": title}

    page_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    metadata = {
        "source": url,
        "source_type": "website",
        "title": title,
        "page_content_hash": page_hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    chunks = chunk_text(text, metadata) 

    added, skipped = _store_chunks(chunks, vector_store, skip_duplicates)

    logger.info("Website %s: %d chunks added, %d skipped (duplicates)", url, added, skipped)
    return {
        "url": url,
        "title": title,
        "chunks_added": added,
        "chunks_skipped": skipped,
        "total_chunks": len(chunks),
    }


def _store_chunks(chunks, vector_store: BaseVectorStore, skip_duplicates: bool) -> tuple[int, int]:
    """Store chunks, optionally skipping ones whose content_hash already exists."""
    added = 0
    skipped = 0
    to_store = []

    for chunk in chunks:
        if skip_duplicates and vector_store.document_exists(chunk.metadata["content_hash"]):
            skipped += 1
            continue
        to_store.append(chunk)

    if to_store:
        ids = [
            f"{compute_content_hash(doc.page_content)}-{doc.metadata.get('chunk_index', 0)}"
            for doc in to_store
        ]
        vector_store.add_documents(to_store, ids=ids)
        added = len(to_store)

    return added, skipped


def recrawl_website(
    url: str,
    vector_store: BaseVectorStore,
    use_playwright: bool = False,
) -> dict:
    """Re-crawl a previously ingested site: delete old chunks for the URL, then re-ingest.

    This supports the 'Re-crawling and updating existing website data' nice-to-have.
    """
    logger.info("Re-crawling website: %s", url)
    vector_store.delete(filter={"source": {"$eq": url}})
    return ingest_website(url, vector_store, use_playwright=use_playwright, skip_duplicates=False)
