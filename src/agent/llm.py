"""
src/agent/llm.py
----------------
Shared cached ChatGroq client factory for LangGraph agent nodes.
"""

import os
from functools import lru_cache
from langchain_groq import ChatGroq


@lru_cache(maxsize=8)
def get_chat_groq(temperature: float = 0.0) -> ChatGroq:
    """
    Return a cached ChatGroq client for the given temperature.

    Reuses client instances across pipeline nodes to avoid repeated construction.
    """
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )
