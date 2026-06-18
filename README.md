# RAG Project

A Retrieval-Augmented Generation (RAG) application built with **FastAPI**,
**LangChain**, and **LangGraph**. It ingests data from websites and uploaded
files (PDF/DOCX/TXT/MD), stores embeddings in a vector database (Pinecone by
default, with a local Chroma fallback), and answers complex, multi-part,
conversational queries via a LangGraph workflow that performs query
decomposition, retrieval, and response aggregation with source citations.

## Architecture

```
                ┌──────────────────────────┐
                │        FastAPI App        │
                │   (app/main.py)            │
                └─────────────┬─────────────┘
                               │
        ┌──────────────────────┴───────────────────────┐
        │                                                │
┌───────▼────────┐                              ┌────────▼─────────┐
│ Ingestion APIs  │                              │   Chat API        │
│ /ingest/url     │                              │   /chat           │
│ /ingest/url/    │                              │                   │
│   recrawl       │                              │  LangGraph        │
│ /ingest/file    │                              │  workflow         │
└───────┬─────────┘                              └────────┬──────────┘
        │                                                  │
┌───────▼─────────────┐                         ┌──────────▼─────────────────────┐
│ Ingestion pipeline   │                         │ analyze_and_decompose_query     │
│ - web_scraper.py     │                         │   -> retrieve                   │
│ - file_processor.py  │                         │   -> generate_response          │
│ - chunking.py        │                         │ (state managed via LangGraph    │
└───────┬──────────────┘                         │  checkpointer for memory)       │
        │                                         └──────────┬─────────────────────┘
        │                                                     │
        └───────────────────► Vector Store ◄─────────────────┘
                          (app/vectorstore/*)
                  BaseVectorStore interface
                  ├── PineconeVectorStore (default)
                  └── ChromaVectorStore (local fallback)
```

### Components

- **`app/main.py`** — FastAPI app, CORS, routes, and a minimal HTML test UI at `/`.
- **`app/core/`** — configuration (`config.py`, via pydantic-settings/.env) and logging setup.
- **`app/ingestion/`**
  - `chunking.py` — shared `RecursiveCharacterTextSplitter`-based chunking and content-hash computation (for duplicate detection).
  - `web_scraper.py` — fetches HTML (via `requests` or optionally Playwright for JS-heavy sites), cleans it with BeautifulSoup, chunks, and stores it. Also supports **re-crawling** (delete old chunks for a URL, then re-ingest).
  - `file_processor.py` — extracts text from PDF (`pypdf`), DOCX (`python-docx`), and TXT/MD files, then chunks and stores it.
- **`app/vectorstore/`**
  - `base.py` — `BaseVectorStore` abstract interface (`add_documents`, `similarity_search`, `delete`, `document_exists`).
  - `pinecone_store.py` — Pinecone serverless implementation (default/preferred).
  - `chroma_store.py` — local Chroma implementation (no external service required), useful for development/testing.
  - `factory.py` — `get_vector_store()` picks the implementation based on `VECTOR_STORE_PROVIDER`. **Swapping vector DBs only requires adding a new adapter class** that implements `BaseVectorStore` and registering it in the factory.
- **`app/graph/`** — the LangGraph workflow:
  - `state.py` — shared `RAGState` TypedDict (conversation messages, original query, decomposed sub-queries + their retrieved chunks, final answer, sources).
  - `nodes.py`:
    - `analyze_and_decompose_query` — uses the LLM + conversation history to resolve references (follow-ups) and split multi-part questions into standalone sub-queries.
    - `retrieve` (factory `make_retrieve_node`) — runs `similarity_search` against the vector store for each sub-query.
    - `generate_response` — synthesizes a single, well-structured answer from all retrieved context, with deduplicated source citations.
  - `workflow.py` — builds and compiles the graph: `START → analyze_and_decompose_query → retrieve → generate_response → END`, using `MemorySaver` as a checkpointer keyed by `thread_id` (= `conversation_id`) for conversation memory.
- **`app/api/`**
  - `schemas.py` — Pydantic request/response models.
  - `routes_ingest.py` — `/ingest/url`, `/ingest/url/recrawl`, `/ingest/file`.
  - `routes_chat.py` — `/chat`.
- **`tests/`** — unit tests for chunking, web scraping/cleaning, file processing, graph nodes (with mocked LLM), and API endpoints (with mocked vector store/graph), using a `FakeVectorStore` so no external services or API keys are needed to run tests.

## Setup

### 1. Prerequisites

- Python 3.10+
- An OpenAI API key (for chat + embedding models)
- A Pinecone API key (if using the default Pinecone vector store)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to use Playwright for JS-heavy website ingestion:

```bash
playwright install chromium
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Key settings:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for chat + embeddings |
| `OPENAI_CHAT_MODEL` | Chat model (default `gpt-4o-mini`) |
| `OPENAI_EMBEDDING_MODEL` | Embedding model (default `text-embedding-3-small`) |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_ENVIRONMENT` | Pinecone region, e.g. `us-east-1` |
| `PINECONE_INDEX_NAME` | Index name (auto-created if it doesn't exist) |
| `VECTOR_STORE_PROVIDER` | `pinecone` (default) or `chroma` (local, no API key needed for the vector DB itself) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Text chunking parameters |

To run locally **without Pinecone**, set `VECTOR_STORE_PROVIDER=chroma`. This stores embeddings in a local Chroma DB under `./data/chroma`.

### 4. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- Minimal test UI: `http://localhost:8000/`

## Usage

### Ingest a website

```bash
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/docs", "use_playwright": false}'
```

### Re-crawl a website (update existing data)

```bash
curl -X POST http://localhost:8000/ingest/url/recrawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/docs"}'
```

### Upload a file

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/document.pdf"
```

Supported file types: `.pdf`, `.docx`, `.txt`, `.md`.

### Chat / query

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
        "message": "What does the documentation say about pricing, and how does it compare to the competitor mentioned earlier?",
        "conversation_id": "user-123"
      }'
```

Response shape:

```json
{
  "answer": "...",
  "sub_queries": [
    {"question": "What does the documentation say about pricing?", "num_chunks_retrieved": 4},
    {"question": "How does the pricing compare to <competitor>?", "num_chunks_retrieved": 4}
  ],
  "sources": [
    {"source": "https://example.com/docs/pricing", "title": "Pricing"}
  ],
  "conversation_id": "user-123"
}
```

`conversation_id` controls conversation memory — send the same ID on follow-up
requests so the workflow can resolve references like "what about its
pricing?" using prior turns.

## How query handling works

1. **`analyze_and_decompose_query`** — the LLM looks at the conversation
   history and the latest message. It rewrites ambiguous follow-ups into
   self-contained questions and splits multi-part messages (e.g. "What's the
   pricing and what are the system requirements?") into separate
   sub-queries. Purely conversational messages (e.g. "thanks!") fall back to
   a single pass-through "sub-query" so the generation step can respond
   naturally without forcing retrieval.
2. **`retrieve`** — for each sub-query, runs `similarity_search` against the
   configured vector store (top-`k`, configurable per request via
   `retrieval_k`).
3. **`generate_response`** — combines all sub-queries and their retrieved
   chunks (plus conversation history) into a single prompt, and asks the LLM
   to produce one coherent answer addressing every part of the original
   message, citing sources naturally. Deduplicated source metadata is
   returned separately in the API response for clean citation display.

## Duplicate content detection

Each chunk gets a `content_hash` (SHA-256 of normalized text) stored as
metadata. Before storing a new chunk, `document_exists(content_hash)` is
checked against the vector store; matching chunks are skipped. This applies
to both website and file ingestion.

## Re-crawling

`POST /ingest/url/recrawl` deletes all existing vectors with
`metadata.source == <url>` (via the vector store's `delete(filter=...)`) and
then re-ingests the page fresh — useful for keeping website-derived content
up to date.

## Swapping the vector store

The rest of the app only depends on `BaseVectorStore`. To add a new backend
(e.g. Weaviate, Qdrant, FAISS):

1. Create `app/vectorstore/<name>_store.py` implementing `BaseVectorStore`.
2. Register it in `app/vectorstore/factory.py` under a new
   `VECTOR_STORE_PROVIDER` value.
3. No changes needed to ingestion, graph, or API code.
