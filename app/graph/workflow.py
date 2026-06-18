"""
Builds the LangGraph workflow for the RAG application.

Workflow:

    START -> analyze_and_decompose_query -> retrieve -> generate_response -> END

Conversation memory is handled via LangGraph's checkpointer (in-memory by
default), keyed by a `thread_id` so multiple users/conversations can be
maintained independently and follow-up questions retain context.
"""
from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import analyze_and_decompose_query, generate_response, make_retrieve_node
from app.graph.state import RAGState
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.factory import get_vector_store


def build_rag_graph(vector_store: BaseVectorStore, retrieval_k: int = 4):
    """Construct and compile the RAG LangGraph workflow."""
    retrieve_node = make_retrieve_node(vector_store, k=retrieval_k)

    graph = StateGraph(RAGState)

    graph.add_node("analyze_and_decompose_query", analyze_and_decompose_query)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "analyze_and_decompose_query")
    graph.add_edge("analyze_and_decompose_query", "retrieve")
    graph.add_edge("retrieve", "generate_response")
    graph.add_edge("generate_response", END)

    # In-memory checkpointer for conversation memory across turns within a thread.
    # For production, swap MemorySaver for a persistent checkpointer
    # (e.g. SqliteSaver, PostgresSaver) without changing any node logic.
    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


@lru_cache
def get_compiled_graph():
    """Singleton compiled graph using the configured vector store."""
    vector_store = get_vector_store()
    return build_rag_graph(vector_store)
