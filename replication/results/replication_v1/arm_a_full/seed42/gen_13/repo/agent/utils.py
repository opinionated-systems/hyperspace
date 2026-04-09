"""
Utility functions for the agent package.

Provides helper functions for common operations across the agent codebase.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_hash(data: Any) -> str:
    """Compute a SHA-256 hash of the given data.
    
    Args:
        data: Any JSON-serializable data
        
    Returns:
        A hex digest string of the hash
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: The text to truncate
        max_length: Maximum length of the output
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_message_history(msg_history: list[dict]) -> str:
    """Format message history for display/logging.
    
    Args:
        msg_history: List of message dicts with 'role' and 'text' keys
        
    Returns:
        Formatted string representation
    """
    lines = []
    for msg in msg_history:
        role = msg.get("role", "unknown")
        text = msg.get("text", "")
        truncated = truncate_text(text, max_length=200)
        lines.append(f"[{role}]: {truncated}")
    return "\n".join(lines)
