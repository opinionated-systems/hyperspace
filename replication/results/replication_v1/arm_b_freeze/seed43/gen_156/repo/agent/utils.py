"""
Utility functions for the agent system.

Common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len characters, adding suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def format_code_block(code: str, language: str = "") -> str:
    """Format code as a markdown code block."""
    return f"```{language}\n{code}\n```"


def extract_code_from_markdown(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The markdown text to extract from
        language: Optional language filter (e.g., 'python', 'json')
    
    Returns:
        List of code block contents
    """
    pattern = r'```(?:' + (re.escape(language) if language else r'\w*') + r')?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\s-]', '_', name)
    safe = re.sub(r'\s+', '_', safe)
    return safe.strip('_')


def count_tokens_approx(text: str) -> int:
    """Approximate token count (very rough estimate: ~4 chars per token)."""
    return len(text) // 4


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width."""
    return textwrap.fill(text, width=width)


def parse_key_value_pairs(text: str) -> dict[str, str]:
    """Parse simple key: value pairs from text.
    
    Handles formats like:
        key1: value1
        key2: value2
    """
    result = {}
    for line in text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result


def safe_json_dumps(obj: Any, indent: int | None = None) -> str:
    """Safely convert object to JSON string, handling common issues."""
    import json
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        return f'{{"error": "JSON serialization failed: {e}"}}'


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_pattern.sub('', text)


def is_valid_python_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return name.isidentifier() and not name[0].isdigit()


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Return singular or plural form based on count."""
    if plural is None:
        plural = singular + 's'
    return singular if count == 1 else plural


def dedent_all(text: str) -> str:
    """Remove common leading whitespace from all lines."""
    lines = text.split('\n')
    if not lines:
        return text
    
    # Find minimum indentation (excluding empty lines)
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return text
    
    min_indent = min(indents)
    return '\n'.join(line[min_indent:] if line.strip() else line for line in lines)


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float | None = None
    
    def __enter__(self) -> "Timer":
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args) -> None:
        import time
        self.elapsed = time.time() - self.start_time
    
    def __str__(self) -> str:
        if self.elapsed is None:
            return f"{self.name}: still running"
        return f"{self.name}: {self.elapsed:.3f}s"
    
    def get_elapsed(self) -> float:
        """Get elapsed time, or 0 if not finished."""
        return self.elapsed if self.elapsed is not None else 0.0


class StructuredLogger:
    """Structured logging utility for consistent log formatting.
    
    Provides JSON-structured logging with timestamps, log levels,
    and contextual information for better observability.
    """
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level
        self._levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    
    def _should_log(self, level: str) -> bool:
        return self._levels.get(level, 20) >= self._levels.get(self.level, 20)
    
    def _format(self, level: str, message: str, **kwargs) -> str:
        import json
        from datetime import datetime, timezone
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": self.name,
            "level": level,
            "message": message,
        }
        if kwargs:
            entry["context"] = kwargs
        return json.dumps(entry, default=str)
    
    def debug(self, message: str, **kwargs) -> None:
        if self._should_log("DEBUG"):
            print(self._format("DEBUG", message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        if self._should_log("INFO"):
            print(self._format("INFO", message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        if self._should_log("WARNING"):
            print(self._format("WARNING", message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        if self._should_log("ERROR"):
            print(self._format("ERROR", message, **kwargs))
