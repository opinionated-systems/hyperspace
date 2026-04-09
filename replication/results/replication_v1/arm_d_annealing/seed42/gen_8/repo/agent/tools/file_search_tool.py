"""
File search tool: search for files containing specific patterns.

Provides grep-like functionality to find files by content,
useful for locating code to modify.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for files containing a specific pattern. Returns matching file paths with line numbers and context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for in file contents",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (default: current directory)",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., '.py', '.js')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (default: 20)",
                    },
                },
                "required": ["pattern"],
            },
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    max_results: int = 20,
) -> str:
    """Search for files containing the given pattern.

    Args:
        pattern: Regex pattern to search for
        path: Directory to search in
        file_extension: Optional extension filter (e.g., '.py')
        max_results: Maximum matches to return

    Returns:
        Formatted string with matches
    """
    matches = []
    compiled_pattern = re.compile(pattern)

    try:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for filename in files:
                if file_extension and not filename.endswith(file_extension):
                    continue

                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        for i, line in enumerate(content.splitlines(), 1):
                            if compiled_pattern.search(line):
                                matches.append({
                                    "file": filepath,
                                    "line": i,
                                    "content": line.strip()[:100],  # Truncate long lines
                                })
                                if len(matches) >= max_results:
                                    break
                        if len(matches) >= max_results:
                            break
                except (IOError, OSError):
                    continue

            if len(matches) >= max_results:
                break

    except Exception as e:
        return f"Error searching: {e}"

    if not matches:
        return f"No matches found for pattern '{pattern}'"

    result_lines = [f"Found {len(matches)} matches for pattern '{pattern}':\n"]
    for m in matches:
        result_lines.append(f"  {m['file']}:{m['line']}: {m['content']}")

    return "\n".join(result_lines)
