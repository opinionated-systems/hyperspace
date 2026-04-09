"""
Search tool: search for text patterns within files.

Provides grep-like functionality to find content across the codebase.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for text patterns within files. Returns matching lines with file paths and line numbers.",
            "parameters": {
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
                        "description": "Optional file extension filter (e.g., '.py', '.js'). Searches all files if not specified.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return. Defaults to 50.",
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
    max_results: int = 50,
) -> dict:
    """Search for pattern in files under path.

    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional extension filter (e.g., '.py')
        max_results: Maximum matches to return

    Returns:
        Dict with matches list and count
    """
    matches = []
    count = 0

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {
            "error": f"Invalid regex pattern: {e}",
            "matches": [],
            "count": 0,
        }

    if os.path.isfile(path):
        files_to_search = [path]
    elif os.path.isdir(path):
        files_to_search = []
        for root, _dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            if "/." in root or "__pycache__" in root:
                continue
            for f in files:
                if file_extension and not f.endswith(file_extension):
                    continue
                files_to_search.append(os.path.join(root, f))
    else:
        return {
            "error": f"Path not found: {path}",
            "matches": [],
            "count": 0,
        }

    for filepath in files_to_search:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append({
                            "file": filepath,
                            "line": line_num,
                            "content": line.rstrip("\n"),
                        })
                        count += 1
                        if count >= max_results:
                            break
        except (IOError, OSError, UnicodeDecodeError):
            continue
        if count >= max_results:
            break

    return {
        "matches": matches,
        "count": count,
        "truncated": count >= max_results,
    }
