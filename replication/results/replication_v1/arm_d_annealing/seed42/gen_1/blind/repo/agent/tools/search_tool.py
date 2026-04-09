"""
Search tool for finding patterns in files.

Provides grep-like functionality to search for text patterns within files.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "search",
        "description": "Search for a pattern in files within a directory. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in. Defaults to current directory.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt'). If not provided, searches all files.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return. Defaults to 50.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for pattern in files and return matches."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    matches = []
    count = 0

    if os.path.isfile(path):
        files_to_search = [path]
    else:
        files_to_search = []
        for root, _, files in os.walk(path):
            for filename in files:
                if file_extension is None or filename.endswith(file_extension):
                    files_to_search.append(os.path.join(root, filename))

    for filepath in files_to_search:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append(f"{filepath}:{line_num}: {line.rstrip()}")
                        count += 1
                        if count >= max_results:
                            break
                    if count >= max_results:
                        break
        except (IOError, OSError) as e:
            continue

    if not matches:
        return f"No matches found for pattern '{pattern}'"

    result = f"Found {len(matches)} match(es) for pattern '{pattern}':\n"
    result += "\n".join(matches)
    if count >= max_results:
        result += f"\n... (truncated to {max_results} results)"
    return result
