"""
API routes for querying the RAG system via the LangGraph workflow.
"""
from fastapi import APIRouter, HTTPException

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from app.api.schemas import ChatRequest, ChatResponse, SourceCitation, SubQueryResult
from app.core.logging import get_logger
from app.graph.workflow import build_rag_graph
from app.vectorstore.factory import get_vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the RAG assistant.

    Conversation memory is keyed by `conversation_id`, so follow-up questions
    within the same conversation retain context.
    """
    vector_store = get_vector_store()
    graph = build_rag_graph(vector_store, retrieval_k=request.retrieval_k)

    config: RunnableConfig = {"configurable": {"thread_id": request.conversation_id}}

    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "original_query": request.message,
                "sub_queries": [],
                "final_answer": None,
                "sources": [],
            },
            config:=config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat graph invocation failed")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {exc}") from exc

    sub_query_results = [
        SubQueryResult(question=sq["question"], num_chunks_retrieved=len(sq["retrieved_chunks"]))
        for sq in result.get("sub_queries", [])
    ]
    sources = [SourceCitation(**src) for src in result.get("sources", [])]

    return ChatResponse(
        answer=result.get("final_answer") or "",
        sub_queries=sub_query_results,
        sources=sources,
        conversation_id=request.conversation_id,
    )
