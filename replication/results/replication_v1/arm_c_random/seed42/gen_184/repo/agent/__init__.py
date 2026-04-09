"""Agent package: tools and agentic loop for self-improvement."""

from __future__ import annotations

from agent.agentic_loop import chat_with_agent
from agent.llm_client import LLMClient, META_MODEL, EVAL_MODEL
from agent import utils

__all__ = ["chat_with_agent", "LLMClient", "META_MODEL", "EVAL_MODEL", "utils"]

