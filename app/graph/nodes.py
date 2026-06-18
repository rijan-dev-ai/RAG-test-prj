"""
Node implementations for the RAG LangGraph workflow:

1. analyze_and_decompose_query - rewrites the query using conversation context
   and breaks complex/multi-part queries into standalone sub-queries.
2. retrieve - for each sub-query, retrieve relevant chunks from the vector store.
3. generate_response - aggregate retrieved context and synthesize a final answer
   with source citations.
"""
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.state import RAGState, SubQuery
from app.vectorstore.base import BaseVectorStore

logger = get_logger(__name__)


def _get_llm(temperature: float = 0.0) -> ChatOpenAI:
    settings = get_settings()

    print("🔥 GROQ KEY LOADED:", bool(settings.groq_api_key))
    print("🔥 MODEL:", settings.groq_chat_model)

    return ChatOpenAI(
        model=settings.groq_chat_model,
        temperature=temperature,
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

DECOMPOSITION_PROMPT = """You are a query analysis assistant for a Retrieval-Augmented Generation system.

Given the conversation history and the user's latest message, do the following:
1. Resolve any pronouns or references using the conversation history so each sub-query is self-contained \
(e.g. "what about its pricing?" -> "What is the pricing of <Product X>?").
2. If the message contains multiple distinct questions or asks for multiple pieces of information, \
split it into separate standalone sub-queries.
3. If the message is a single simple question, return it as a single sub-query (rewritten to be \
self-contained if needed).
4. If the message is conversational and does not require document retrieval (e.g. "thanks", "hello"), \
return an empty list.

Respond with ONLY a JSON object in this exact format, no markdown fences, no extra text:
{{"sub_queries": ["question 1", "question 2", ...]}}

Conversation history:
{history}

Latest user message:
{query}
"""


def analyze_and_decompose_query(state: RAGState) -> dict[str, Any]:
    """Analyze the latest user message and decompose it into standalone sub-queries."""
    query = state["original_query"]

    # Build a short textual history (excluding the current message) for context resolution
    history_messages = state["messages"][:-1] if state["messages"] else []
    history_lines = []
    for msg in history_messages[-10:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines) if history_lines else "(no previous turns)"

    llm = _get_llm(temperature=0.0)
    prompt = DECOMPOSITION_PROMPT.format(history=history_text, query=query)

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content

    sub_query_texts = _parse_sub_queries(raw, fallback=query)

    sub_queries: list[SubQuery] = [
        {"question": q, "retrieved_chunks": []} for q in sub_query_texts
    ]
    
    logger.info("Decomposed query into %d sub-queries: %s", len(sub_queries), sub_query_texts)
    return {"sub_queries": sub_queries}


def _parse_sub_queries(raw: str, fallback: str) -> list[str]:
    """Parse the LLM's JSON response, with a safe fallback to the original query."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        sub_queries = data.get("sub_queries", [])
        sub_queries = [q.strip() for q in sub_queries if isinstance(q, str) and q.strip()]
        if sub_queries:
            return sub_queries
        # Empty list -> conversational message, no retrieval needed; keep one entry
        # so downstream nodes can still respond conversationally.
        return [fallback] if not sub_queries else sub_queries
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse decomposition JSON, falling back to original query")
        return [fallback]


def make_retrieve_node(vector_store: BaseVectorStore, k: int = 4):
    """Factory returning a retrieve node bound to a specific vector store instance."""

    def retrieve(state: RAGState) -> dict[str, Any]:
        updated_sub_queries: list[SubQuery] = []

        for sub_query in state["sub_queries"]:
            question = sub_query["question"]
            try:
                docs = vector_store.similarity_search(question, k=k)
            except Exception as exc:  # noqa: BLE001
                logger.error("Retrieval failed for sub-query '%s': %s", question, exc)
                docs = []

            retrieved_chunks = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", ""),
                    "score": doc.metadata.get("score"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                }
                for doc in docs
            ]

            updated_sub_queries.append({"question": question, "retrieved_chunks": retrieved_chunks})

        logger.info(
            "Retrieved chunks for %d sub-queries (total chunks: %d)",
            len(updated_sub_queries),
            sum(len(sq["retrieved_chunks"]) for sq in updated_sub_queries),
        )
        return {"sub_queries": updated_sub_queries}

    return retrieve


GENERATION_PROMPT = """You are a helpful assistant answering questions using retrieved context.

Conversation history:
{history}

The user's original message was:
{original_query}

It was broken down into the following sub-questions, each with retrieved context:

{context_blocks}

Instructions:
- Answer the user's original message fully, addressing each sub-question.
- Use ONLY the retrieved context to answer factual questions. If the context does not contain \
the answer, say so honestly rather than making something up.
- Write a single, well-structured, coherent response (do not just list answers per sub-question \
unless that structure makes sense for the user's request).
- When you use information from a source, mention it naturally (e.g., "According to [source]...").
- If the message was conversational (e.g. a greeting) and no context was retrieved, respond naturally.
"""


def generate_response(state: RAGState) -> dict[str, Any]:
    """Aggregate retrieved context across sub-queries and synthesize the final answer."""
    history_messages = state["messages"][:-1] if state["messages"] else []
    history_lines = []
    for msg in history_messages[-10:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines) if history_lines else "(no previous turns)"

    context_blocks = []
    sources: list[dict[str, Any]] = []
    seen_sources = set()

    for sub_query in state["sub_queries"]:
        chunks = sub_query["retrieved_chunks"]
        if chunks:
            chunk_texts = []
            for chunk in chunks:
                chunk_texts.append(
                    f"(Source: {chunk['source']}) {chunk['content']}"
                )
                source_key = (chunk["source"], chunk.get("title", ""))
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources.append({"source": chunk["source"], "title": chunk.get("title", "")})
            context_block = "\n---\n".join(chunk_texts)
        else:
            context_block = "(no relevant context retrieved)"

        context_blocks.append(f"Sub-question: {sub_query['question']}\nContext:\n{context_block}")

    prompt = GENERATION_PROMPT.format(
        history=history_text,
        original_query=state["original_query"],
        context_blocks="\n\n".join(context_blocks),
    )

    llm = _get_llm(temperature=0.2)
    response = llm.invoke([HumanMessage(content=prompt)])
    final_answer = response.content.strip()

    logger.info("Generated final answer (%d chars), %d unique sources", len(final_answer), len(sources))

    return {
        "final_answer": final_answer,
        "sources": sources,
        "messages": [AIMessage(content=final_answer)],
    }
