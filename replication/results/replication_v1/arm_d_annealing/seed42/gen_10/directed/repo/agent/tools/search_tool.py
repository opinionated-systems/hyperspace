"""
Search tool: find files and search content within the codebase.

Provides grep-like file content search and file finding capabilities.
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
            "name": "search",
            "description": "Search for files or content within the codebase. Supports grep-like pattern matching and file finding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex for content search, glob for file search)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Defaults to current directory.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["content", "files"],
                        "description": "Type of search: 'content' searches file contents, 'files' searches filenames",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., '.py', '.md')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 50)",
                    },
                },
                "required": ["pattern", "search_type"],
            },
        },
    }


def tool_function(
    pattern: str,
    search_type: str,
    path: str = ".",
    file_extension: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """Execute search operation.

    Args:
        pattern: Search pattern
        search_type: 'content' or 'files'
        path: Directory or file to search
        file_extension: Optional extension filter
        max_results: Maximum results to return

    Returns:
        Dict with search results or error info
    """
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return {
                "success": False,
                "error": f"Path does not exist: {path}",
                "results": [],
            }

        results = []

        if search_type == "files":
            # Find files by name pattern
            results = _find_files(pattern, path, file_extension, max_results)
        elif search_type == "content":
            # Search file contents
            results = _search_content(pattern, path, file_extension, max_results)
        else:
            return {
                "success": False,
                "error": f"Unknown search_type: {search_type}",
                "results": [],
            }

        return {
            "success": True,
            "count": len(results),
            "results": results[:max_results],
            "truncated": len(results) > max_results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": [],
        }


def _find_files(
    pattern: str,
    path: str,
    file_extension: str | None,
    max_results: int,
) -> list[dict]:
    """Find files matching pattern."""
    results = []
    pattern_lower = pattern.lower()

    for root, dirs, files in os.walk(path):
        # Skip hidden directories and common non-source dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]

        for filename in files:
            if file_extension and not filename.endswith(file_extension):
                continue

            if pattern_lower in filename.lower() or re.search(pattern, filename):
                full_path = os.path.join(root, filename)
                results.append({
                    "path": full_path,
                    "type": "file",
                })

            if len(results) >= max_results:
                return results

    return results


def _search_content(
    pattern: str,
    path: str,
    file_extension: str | None,
    max_results: int,
) -> list[dict]:
    """Search file contents using grep-like matching."""
    results = []

    # Build file list
    files_to_search = []
    if os.path.isfile(path):
        files_to_search = [path]
    else:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]
            for filename in files:
                if file_extension and not filename.endswith(file_extension):
                    continue
                files_to_search.append(os.path.join(root, filename))

    # Search each file
    compiled_pattern = re.compile(pattern, re.IGNORECASE)

    for filepath in files_to_search:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if compiled_pattern.search(line):
                        results.append({
                            "path": filepath,
                            "line": line_num,
                            "content": line.rstrip(),
                            "type": "match",
                        })
                        if len(results) >= max_results:
                            return results
        except (IOError, OSError):
            continue

    return results
