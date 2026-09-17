"""
src/agent/nodes/generator.py
-----------------------------
The Generator node calls the Groq API (llama-3.3-70b-versatile) with the retrieved
legislative sections and produces a cited answer to the user's question.

CITATION ENFORCEMENT:
The system prompt strictly requires every answer to cite specific CSPA
sections (e.g. "CSPA s.11(1)"). The model is instructed to:
1. Only answer from the provided sections — no hallucination
2. Always cite the specific section for every claim
3. Clearly state when the answer cannot be found in the provided sections

This is the most critical requirement for legislative oversight QA.

"""

import logging
from src.agent.llm import get_chat_groq
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """You are an AI Quality Assurance and Legal Compliance assistant specialized in Ontario Police Oversight Legislation (CSPA 2019 & Regulations).

You answer questions about Ontario police oversight based ONLY on the legislative 
sections provided below. You are an expert at translating legislation into 
plain-language QA expectations.

STRICT RULES:
1. Every claim you make MUST cite the specific section: e.g. "CSPA s.11(1)" or "CSPA s.41"
2. NEVER answer from your own knowledge — only use the provided sections
3. If the provided sections do not contain enough information to answer, say:
   "The provided sections do not address this question directly. 
    Please consult [relevant section topic] for more information."
4. When asked for a checklist or lines of inquiry, format as a numbered list
5. Keep answers clear and professional — this is for legal and oversight QA professionals

FORMAT:
- Start with a direct answer to the question
- Support each point with a citation in brackets: [CSPA s.X(Y)]
- End with a "Sources" section listing all cited sections"""


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured string for the LLM prompt.

    Each chunk is formatted with its citation header so the model can
    easily reference it in the answer.

    Args:
        chunks: List of result dicts from the hybrid retriever.

    Returns:
        Formatted string with all chunks, ready to include in the prompt.
    """
    formatted_sections = []
    for i, chunk in enumerate(chunks, start=1):
        citation = chunk.get("citation", f"Section {i}")
        title = chunk.get("section_title", "")
        text = chunk["text"]

        section = f"[{citation} — {title}]\n{text}"
        formatted_sections.append(section)

    return "\n\n---\n\n".join(formatted_sections)


def generate_answer(state: AgentState) -> AgentState:
    """
    Generate a cited answer using the retrieved legislative sections.

    Formats the retrieved chunks as context, then calls Groq with a
    strict system prompt that enforces citation requirements.

    Args:
        state: AgentState with question and retrieved_chunks populated.

    Returns:
        Updated AgentState with "answer" field set.
    """
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])

    # Off-topic / conversational path — no chunks, no retrieval
    if state.get("route") == "conversational":
        return {
            "answer": "I'm designed to answer questions about Ontario police oversight legislation (CSPA, O. Regs, and LECA). I'm not able to help with that question.",
            "confidence_label": None,
            "confidence_score": 0.0,
        }

    logger.info(f"[generator] Generating answer for: '{question[:60]}'")

    # Format the retrieved sections for the prompt
    sections_text = format_chunks_for_prompt(chunks)

    # Build the full user message with context + question
    user_message = f"""LEGISLATIVE SECTIONS:
{sections_text}

---

QUESTION: {question}

Please answer based only on the sections provided above, with citations."""

    llm = get_chat_groq(temperature=0.1)

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info(f"[generator] Answer generated ({len(answer)} chars).")

    return {"answer": answer,
            "confidence_label": state.get("confidence_label"),
            "confidence_score": state.get("confidence_score")
        }
