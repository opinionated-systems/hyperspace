"""
Thinking tool: record structured reasoning steps during agent execution.

Provides a way for the meta-agent to explicitly document its thought process,
which improves transparency, debugging, and allows for better analysis
of the agent's decision-making.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# In-memory storage for thinking steps
_thinking_history: list[dict] = []
_max_history_size: int = 1000


def tool_info() -> dict:
    return {
        "name": "thinking",
        "description": (
            "Record a structured thinking step during agent execution. "
            "Use this to document your reasoning, plans, observations, or decisions. "
            "This improves transparency and helps with debugging."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The main thought or reasoning step to record.",
                },
                "thought_type": {
                    "type": "string",
                    "enum": ["observation", "plan", "reasoning", "decision", "reflection", "hypothesis"],
                    "description": "Category of the thought for better organization.",
                },
                "context": {
                    "type": "object",
                    "description": "Optional additional context (key-value pairs) to attach to this thought.",
                },
            },
            "required": ["thought", "thought_type"],
        },
    }


def tool_function(
    thought: str,
    thought_type: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Record a thinking step.
    
    Args:
        thought: The main thought or reasoning to record
        thought_type: Category of thought (observation, plan, reasoning, decision, reflection, hypothesis)
        context: Optional additional context as key-value pairs
    
    Returns:
        Confirmation message with thought ID
    """
    global _thinking_history
    
    # Validate thought_type
    valid_types = ["observation", "plan", "reasoning", "decision", "reflection", "hypothesis"]
    if thought_type not in valid_types:
        return f"Error: Invalid thought_type '{thought_type}'. Must be one of: {valid_types}"
    
    # Create thinking entry
    entry = {
        "id": len(_thinking_history) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thought_type": thought_type,
        "thought": thought,
        "context": context or {},
    }
    
    # Add to history
    _thinking_history.append(entry)
    
    # Trim history if it gets too large
    if len(_thinking_history) > _max_history_size:
        _thinking_history = _thinking_history[-_max_history_size:]
    
    # Log the thought
    logger.info(f"[{thought_type.upper()}] {thought[:100]}...")
    
    return f"Thought recorded (ID: {entry['id']}, Type: {thought_type})"


def get_thinking_history(
    thought_type: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Get the thinking history, optionally filtered by type.
    
    Args:
        thought_type: Optional filter by thought type
        limit: Maximum number of entries to return
    
    Returns:
        List of thinking entries
    """
    history = _thinking_history
    
    if thought_type:
        history = [h for h in history if h["thought_type"] == thought_type]
    
    return history[-limit:]


def clear_thinking_history() -> None:
    """Clear the thinking history."""
    global _thinking_history
    _thinking_history = []


def export_thinking_history(path: str) -> str:
    """Export thinking history to a JSON file.
    
    Args:
        path: Absolute path to save the JSON file
    
    Returns:
        Success or error message
    """
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with open(p, "w") as f:
            json.dump(_thinking_history, f, indent=2, default=str)
        
        return f"Thinking history exported to {path} ({len(_thinking_history)} entries)"
    except Exception as e:
        return f"Error exporting thinking history: {e}"


def get_thinking_summary() -> dict:
    """Get a summary of thinking history statistics."""
    if not _thinking_history:
        return {"total_thoughts": 0, "by_type": {}}
    
    by_type = {}
    for entry in _thinking_history:
        t = entry["thought_type"]
        by_type[t] = by_type.get(t, 0) + 1
    
    return {
        "total_thoughts": len(_thinking_history),
        "by_type": by_type,
        "first_timestamp": _thinking_history[0]["timestamp"],
        "last_timestamp": _thinking_history[-1]["timestamp"],
    }
