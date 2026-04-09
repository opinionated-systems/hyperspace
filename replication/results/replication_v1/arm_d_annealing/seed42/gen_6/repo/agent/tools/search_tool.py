"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality and file finding capabilities.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files or content within the codebase. Supports grep-like pattern matching and file finding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["grep", "find"],
                        "description": "Search command: 'grep' to search file contents, 'find' to search for files by name",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex for grep, glob pattern for find)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (default: current directory)",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., '.py', '.txt')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50)",
                    },
                },
                "required": ["command", "pattern"],
            },
        },
    }


def tool_function(
    command: str,
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    max_results: int = 50,
    **kwargs: Any,
) -> str:
    """Execute search command.

    Args:
        command: 'grep' or 'find'
        pattern: search pattern
        path: directory to search in
        file_extension: optional file extension filter
        max_results: maximum results to return

    Returns:
        Search results as formatted string
    """
    if command == "grep":
        return _grep_search(pattern, path, file_extension, max_results)
    elif command == "find":
        return _find_files(pattern, path, max_results)
    else:
        return f"Error: Unknown command '{command}'. Use 'grep' or 'find'."


def _grep_search(
    pattern: str,
    path: str,
    file_extension: str | None,
    max_results: int,
) -> str:
    """Search file contents using grep."""
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a valid directory."

    results = []
    count = 0

    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for filename in files:
            if file_extension and not filename.endswith(file_extension):
                continue
            if filename.startswith("."):
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            results.append(f"{filepath}:{line_num}:{line.rstrip()}")
                            count += 1
                            if count >= max_results:
                                break
            except (IOError, OSError):
                continue

            if count >= max_results:
                break
        if count >= max_results:
            break

    if not results:
        return f"No matches found for pattern '{pattern}' in '{path}'."

    header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
    return header + "\n".join(results)


def _find_files(
    pattern: str,
    path: str,
    max_results: int,
) -> str:
    """Find files by name pattern."""
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a valid directory."

    results = []
    count = 0

    # Convert glob pattern to regex
    regex_pattern = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")

    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for filename in files:
            if re.search(regex_pattern, filename):
                results.append(os.path.join(root, filename))
                count += 1
                if count >= max_results:
                    break
        if count >= max_results:
            break

    if not results:
        return f"No files found matching pattern '{pattern}' in '{path}'."

    header = f"Found {len(results)} file(s) matching '{pattern}':\n"
    return header + "\n".join(results)
