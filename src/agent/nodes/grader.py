"""
src/agent/nodes/grader.py
--------------------------
The Relevance Grader node checks whether the retrieved chunks are
actually relevant to the user's question before passing them to the
generator.

WHY A GRADER:
The retriever returns the top-k chunks by similarity score, but similarity
doesn't guarantee relevance. A question about "police chief duties" might
retrieve chunks about "municipal board duties" that are topically related
but don't actually answer the question.

The grader asks Groq (llama-3.3-70b-versatile) to evaluate each chunk and decides:
- If enough chunks are relevant → pass to generator
- If not enough chunks are relevant → rewrite the query and retry

This is the core of the CRAG (Corrective RAG) pattern.

Branch: feature/langgraph-agent
Issue:  #9 — Relevance Grader and Query Rewriter nodes
"""

import os
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

GRADER_SYSTEM_PROMPT = """You are a relevance grader for a police oversight QA system.

Given a user question and a legislative section, decide if the section is
relevant to answering the question.

Respond with ONLY "yes" if the section is relevant, or "no" if it is not.
Do not explain. Just yes or no."""


def grade_chunks(state: AgentState) -> AgentState:
    """
    Grade each retrieved chunk for relevance to the question.

    Sets relevance_passed=True if at least 2 chunks are relevant,
    otherwise sets relevance_passed=False to trigger query rewriting.

    Args:
        state: AgentState with question and retrieved_chunks populated.

    Returns:
        Updated AgentState with relevance_passed set.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    logger.info(f"[grader] Grading {len(chunks)} chunks for relevance.")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    relevant_count = 0
    min_relevant = 2  # require at least 2 relevant chunks to pass

    for chunk in chunks:
        # Format the chunk with its citation for context
        chunk_text = f"[{chunk.get('citation', 'Unknown')}]\n{chunk['text'][:500]}"

        messages = [
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nSection: {chunk_text}"),
        ]

        response = llm.invoke(messages)
        verdict = response.content.strip().lower()

        if verdict == "yes":
            relevant_count += 1

        # Early exit if we already have enough relevant chunks
        if relevant_count >= min_relevant:
            break

    relevance_passed = relevant_count >= min_relevant
    logger.info(
        f"[grader] {relevant_count}/{len(chunks)} relevant. "
        f"Passed: {relevance_passed}"
    )
    print(f"[grader] {relevant_count} relevant chunks found. Passed: {relevance_passed}")

    return {"relevance_passed": relevance_passed}
