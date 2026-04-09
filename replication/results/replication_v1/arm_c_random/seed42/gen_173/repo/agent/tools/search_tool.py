"""
Search tool: find text patterns in files using grep-like functionality.

Provides content search capabilities to complement the file and editor tools.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata matching the paper's tool schema."""
    return {
        "name": "search",
        "description": "Search for text patterns in files. Supports regex and recursive directory search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Default: current directory",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in directories. Default: True",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search. Default: False",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for text patterns in files.

    Args:
        pattern: The search pattern (regex supported)
        path: File or directory to search in
        recursive: Search recursively in directories
        case_sensitive: Case-sensitive search
        max_results: Maximum number of results to return

    Returns:
        Formatted search results or error message
    """
    if not pattern:
        return "Error: pattern is required"

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results = []
    result_count = 0

    if os.path.isfile(path):
        # Search single file
        files_to_search = [path]
    elif os.path.isdir(path):
        # Search directory
        if recursive:
            files_to_search = []
            for root, _, files in os.walk(path):
                for filename in files:
                    if not filename.endswith(".pyc") and "__pycache__" not in root:
                        files_to_search.append(os.path.join(root, filename))
        else:
            files_to_search = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
                and not f.endswith(".pyc")
            ]
    else:
        return f"Error: path '{path}' does not exist"

    for filepath in files_to_search:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if compiled_pattern.search(line):
                        results.append(f"{filepath}:{line_num}:{line.rstrip()}")
                        result_count += 1
                        if result_count >= max_results:
                            break
            if result_count >= max_results:
                break
        except Exception as e:
            continue

    if not results:
        return f"No matches found for pattern '{pattern}' in '{path}'"

    output = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
    output += "\n".join(results)
    if result_count >= max_results:
        output += f"\n... (truncated to {max_results} results)"

    return output
