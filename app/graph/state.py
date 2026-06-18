"""
Shared state object passed between nodes of the LangGraph RAG workflow.
"""
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SubQuery(TypedDict):
    question: str
    retrieved_chunks: list[dict[str, Any]]


class RAGState(TypedDict):
    """The state that flows through the LangGraph graph.

    - messages: full conversation history (managed via add_messages reducer)
    - original_query: the raw user input for this turn (may contain multiple questions)
    - sub_queries: decomposed list of standalone sub-questions with their retrieved chunks
    - final_answer: the synthesized response returned to the user
    - sources: deduplicated list of source citations used in the final answer
    """

    messages: Annotated[list[BaseMessage], add_messages]
    original_query: str
    sub_queries: list[SubQuery]
    final_answer: Optional[str]
    sources: list[dict[str, Any]]
