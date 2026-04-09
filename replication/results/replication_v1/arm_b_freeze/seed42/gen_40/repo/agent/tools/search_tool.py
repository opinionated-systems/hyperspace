"""
Search tool for finding files by content pattern.

Provides grep-like functionality to search within files.
"""

from __future__ import annotations

import os
import re


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "search_files",
        "description": (
            "Search for files containing a regex pattern. "
            "Returns file paths with line numbers and matching lines. "
            "Useful for finding code patterns across the codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search recursively",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for in file contents",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py')",
                },
            },
            "required": ["path", "pattern"],
        },
    }


def tool_function(path: str, pattern: str, file_extension: str | None = None) -> str:
    """Search for files containing a pattern.

    Args:
        path: Directory path to search recursively
        pattern: Regex pattern to search for
        file_extension: Optional file extension filter (e.g., '.py')

    Returns:
        String with matching files and line numbers
    """
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a valid directory"

    matches = []
    try:
        compiled_pattern = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    for root, _, files in os.walk(path):
        for filename in files:
            if file_extension and not filename.endswith(file_extension):
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled_pattern.search(line):
                            matches.append(f"{filepath}:{line_num}: {line.rstrip()}")
                            if len(matches) >= 100:  # Limit results
                                break
                    if len(matches) >= 100:
                        break
            except Exception:
                continue
        if len(matches) >= 100:
            break

    if not matches:
        return "No matches found."

    result = "\n".join(matches[:50])  # Return first 50 matches
    if len(matches) > 50:
        result += f"\n... ({len(matches) - 50} more matches)"
    return result
