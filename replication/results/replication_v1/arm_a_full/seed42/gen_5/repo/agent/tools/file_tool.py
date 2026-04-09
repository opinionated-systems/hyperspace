"""
File search tool: grep and find files in the codebase.

Provides file search capabilities to help agents explore large codebases.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": (
                "Search for files or content within files in a directory. "
                "Supports grep-style pattern matching and file finding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["grep", "find", "glob"],
                        "description": "Search command to execute",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (regex for grep, glob pattern for glob)",
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
) -> str:
    """Execute file search command.

    Args:
        command: One of 'grep', 'find', 'glob'
        pattern: Search pattern
        path: Directory to search in
        file_extension: Optional extension filter
        max_results: Maximum results to return

    Returns:
        Search results as formatted string
    """
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a valid directory"

    results: list[str] = []

    try:
        if command == "grep":
            results = _grep_search(path, pattern, file_extension, max_results)
        elif command == "find":
            results = _find_files(path, pattern, max_results)
        elif command == "glob":
            results = _glob_search(path, pattern, max_results)
        else:
            return f"Error: Unknown command '{command}'"
    except Exception as e:
        return f"Error executing search: {e}"

    if not results:
        return "No results found."

    return "\n".join(results[:max_results])


def _grep_search(
    path: str, pattern: str, file_extension: str | None, max_results: int
) -> list[str]:
    """Search file contents using regex pattern."""
    results = []
    regex = re.compile(pattern, re.IGNORECASE)

    for root, _, files in os.walk(path):
        # Skip hidden directories and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in root.split(os.sep)):
            continue

        for filename in files:
            if file_extension and not filename.endswith(file_extension):
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            rel_path = os.path.relpath(filepath, path)
                            results.append(f"{rel_path}:{i}: {line.rstrip()}")
                            if len(results) >= max_results:
                                return results
            except (IOError, OSError):
                continue

    return results


def _find_files(path: str, pattern: str, max_results: int) -> list[str]:
    """Find files by name pattern."""
    results = []
    regex = re.compile(pattern, re.IGNORECASE)

    for root, _, files in os.walk(path):
        # Skip hidden directories
        if any(part.startswith(".") or part == "__pycache__" for part in root.split(os.sep)):
            continue

        for filename in files:
            if regex.search(filename):
                rel_path = os.path.relpath(os.path.join(root, filename), path)
                results.append(rel_path)
                if len(results) >= max_results:
                    return results

    return results


def _glob_search(path: str, pattern: str, max_results: int) -> list[str]:
    """Find files using glob pattern."""
    import fnmatch

    results = []

    for root, _, files in os.walk(path):
        # Skip hidden directories
        if any(part.startswith(".") or part == "__pycache__" for part in root.split(os.sep)):
            continue

        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                rel_path = os.path.relpath(os.path.join(root, filename), path)
                results.append(rel_path)
                if len(results) >= max_results:
                    return results

    return results
