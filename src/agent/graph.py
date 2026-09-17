"""
src/agent/graph.py
------------------
Defines and compiles the LangGraph StateGraph for the Police Oversight CRAG pipeline.

GRAPH STRUCTURE (CRAG pattern):
                    
    START
      │
      ▼
   [Router] ──── conversational ────────────────────┐
      │                                              │
   retrieval                                         │
      │                                              │
      ▼                                              │
  [Retriever]                                        │
      │                                              │
      ▼                                              │
  [Reranker]                                         │
      │                                              │
   [Grader] ──── pass ──────────────────────────────┤
      │                                              │
    fail                                             │
      │                                              ▼
  [Rewriter] ──────────────────────────────────[Generator]
      │                                              │
      └──── retry (back to Retriever) ───────────────┘
                                                     │
                                                    END

The graph is compiled once at startup and reused for all queries.
LangGraph handles the routing between nodes based on the edge functions.

"""

import logging
from functools import partial
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.router import route_question
from src.agent.nodes.retriever import retrieve_chunks
from src.agent.nodes.reranker import rerank_chunks
from src.agent.nodes.grader import grade_chunks
from src.agent.nodes.rewriter import rewrite_query, MAX_REWRITES
from src.agent.nodes.generator import generate_answer
from src.ingestion.chunker import LegislativeChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge functions — decide which node to go to next
# ---------------------------------------------------------------------------

def route_decision(state: AgentState) -> str:
    """
    Edge function after Router node.
    Returns the name of the next node based on the route decision.
    """
    route = state.get("route", "retrieval")
    if route == "conversational":
        return "generator"
    return "retriever"


def grader_decision(state: AgentState) -> str:
    """
    Edge function after Grader node.
    Returns "generator" if relevance passed, "rewriter" if not.
    """
    if state.get("relevance_passed", False):
        return "generator"
    return "rewriter"


def rewriter_decision(state: AgentState) -> str:
    """
    Edge function after Rewriter node.
    Returns "retriever" to retry, or "generator" if max rewrites reached.
    """
    # If rewriter forced relevance_passed=True, go straight to generator
    if state.get("relevance_passed", False):
        return "generator"
    return "retriever"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(chunks: list[LegislativeChunk]):
    """
    Build and compile the LangGraph StateGraph.

    The graph is built with all nodes and edges defined. The chunks list
    is passed in so the retriever node can build its BM25 index.

    We use functools.partial to bind the chunks argument to the retriever
    node function, since LangGraph node functions only receive state.

    Args:
        chunks: Full list of LegislativeChunk objects from the CSPA document.
                Used to initialise the HybridRetriever's BM25 index.

    Returns:
        A compiled LangGraph graph ready to invoke with a question.

    Example:
        >>> graph = build_graph(chunks)
        >>> result = graph.invoke({"question": "What are police duties?"})
        >>> print(result["answer"])
    """
    # Create the graph with our AgentState schema
    workflow = StateGraph(AgentState)

    # Bind chunks to the retriever node so it can build the BM25 index.
    # partial() creates a new function with chunks pre-filled.
    retriever_node = partial(retrieve_chunks, chunks=chunks)

    # Add all nodes to the graph
    workflow.add_node("router", route_question)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reranker", rerank_chunks)
    workflow.add_node("grader", grade_chunks)
    workflow.add_node("rewriter", rewrite_query)
    workflow.add_node("generator", generate_answer)

    # Set the entry point
    workflow.set_entry_point("router")

    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever": "retriever",
            "generator": "generator",
        }
    )

    # Retriever → Reranker → Grader (Stage 3 precision filter inserted here)
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "grader")

    # Grader conditionally goes to generator or rewriter
    workflow.add_conditional_edges(
        "grader",
        grader_decision,
        {
            "generator": "generator",
            "rewriter": "rewriter",
        }
    )

    # Rewriter conditionally retries or forces to generator
    workflow.add_conditional_edges(
        "rewriter",
        rewriter_decision,
        {
            "retriever": "retriever",
            "generator": "generator",
        }
    )

    # Generator is the terminal node
    workflow.add_edge("generator", END)

    # Compile and return the graph
    graph = workflow.compile()

    logger.info("[graph] LangGraph agent compiled.")

    return graph


def run_query(graph, question: str) -> dict:
    """
    Run a single question through the compiled graph.

    Args:
        graph:    Compiled LangGraph graph from build_graph().
        question: The user's question as a plain string.

    Returns:
        The final AgentState dict with the answer and all intermediate state.

    Example:
        >>> result = run_query(graph, "What are the duties of a police chief?")
        >>> print(result["answer"])
    """
    # Initialise the state with default values
    initial_state: AgentState = {
        "question": question,
        "retrieved_chunks": [],
        "relevance_passed": None,
        "rewrite_count": 0,
        "answer": None,
        "route": None,
        "confidence_label": None,
        "confidence_score": 0.0,
    }

    result = graph.invoke(initial_state)
    return {
        "answer": result.get("answer"),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "confidence": result.get("confidence_label", "Low"),
        "confidence_score": result.get("confidence_score", 0.0),
        "route": result.get("route"),
    }
