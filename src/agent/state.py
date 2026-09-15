"""
src/agent/state.py
------------------
Defines the AgentState TypedDict that flows through the LangGraph graph.

WHY A TYPED STATE:
LangGraph passes a single state object between all nodes. Using a TypedDict
makes the state explicit and type-safe — every node knows exactly what fields
are available and what type they are. This prevents bugs from typos or
missing fields that would otherwise only surface at runtime.

Branch: feature/langgraph-agent
Issue:  #8 — LangGraph graph definition and Router node
"""

from typing import TypedDict, Optional
from typing import Annotated

def take_latest(current, updated):
    return updated if updated else current

class AgentState(TypedDict):
    """
    The state object that flows through every node in the LangGraph graph.

    Each node reads from this state and returns a dict with the fields
    it wants to update. LangGraph merges those updates into the state
    automatically before passing it to the next node.

    Fields:
        question:         The user's original question, unchanged throughout.
        retrieved_chunks: List of chunk dicts returned by the hybrid retriever.
                          Empty list until the retriever node runs.
        relevance_passed: True if the grader found enough relevant chunks.
                          None until the grader node runs.
        rewrite_count:    Number of times the query has been rewritten.
                          Used to prevent infinite retry loops (max 2).
        answer:           The final generated answer with citations.
                          None until the generator node runs.
        route:            Either "retrieval" or "conversational" — set by router.
    """
    question: str
    retrieved_chunks: list[dict]
    relevance_passed: Optional[bool]
    rewrite_count: int
    answer: Optional[str]
    route: Optional[str]
    confidence_label: Annotated[str, take_latest]
    confidence_score: Annotated[float, take_latest]
    rrf_scores: list[float]