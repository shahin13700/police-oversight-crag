"""
src/agent/nodes/rewriter.py
----------------------------
The Query Rewriter node rewrites the user's question to be more
specific and legislative in phrasing when the grader fails.

WHY REWRITING:
Users often ask questions in casual language that doesn't match the
formal language of legislation. "What can police do wrong?" won't
retrieve chunks about "misconduct" and "complaints procedures".

The rewriter translates casual questions into legislative language
to improve retrieval quality on the retry.

MAX RETRIES:
We cap rewrites at 2 to prevent infinite loops. After 2 failed
rewrites, we pass whatever chunks we have to the generator anyway
with a note that the answer may be incomplete.

Branch: feature/langgraph-agent
Issue:  #9 — Relevance Grader and Query Rewriter nodes
"""

import os
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_REWRITES = 2

REWRITER_SYSTEM_PROMPT = """You are a query rewriter for a police oversight legislation QA system.

The original query failed to retrieve relevant sections from the
Community Safety and Policing Act (CSPA) of Ontario.

Rewrite the query to:
1. Use more formal, legislative language
2. Reference specific legal concepts (e.g. "duties", "obligations", "powers")
3. Be more specific about which aspect of policing is being asked about

Return ONLY the rewritten query. No explanation, no preamble."""


def rewrite_query(state: AgentState) -> AgentState:
    """
    Rewrite the user's question using more legislative language.

    Increments rewrite_count. If max rewrites reached, sets
    relevance_passed=True to force the generator to proceed anyway.

    Args:
        state: Current AgentState with the original question.

    Returns:
        Updated AgentState with rewritten question and incremented count.
    """
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    # Check if we've hit the max rewrite limit
    if rewrite_count >= MAX_REWRITES:
        logger.warning(
            f"[rewriter] Max rewrites ({MAX_REWRITES}) reached. "
            "Forcing generator with available chunks."
        )
        print(f"[rewriter] Max rewrites reached — proceeding with available chunks.")
        # Force relevance_passed=True so we don't loop forever
        return {"relevance_passed": True, "rewrite_count": rewrite_count}

    logger.info(f"[rewriter] Rewriting query (attempt {rewrite_count + 1}): '{question[:60]}'")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )

    messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=f"Original query: {question}"),
    ]

    response = llm.invoke(messages)
    rewritten = response.content.strip()

    logger.info(f"[rewriter] Rewritten to: '{rewritten[:60]}'")
    print(f"[rewriter] '{question[:40]}' → '{rewritten[:40]}'")

    return {
        "question": rewritten,
        "rewrite_count": rewrite_count + 1,
        "relevance_passed": None,   # reset so grader runs again
        "retrieved_chunks": [],     # reset so retriever runs again
    }
