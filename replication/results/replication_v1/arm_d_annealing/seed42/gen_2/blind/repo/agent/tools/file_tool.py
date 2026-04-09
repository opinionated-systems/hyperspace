"""
File search tool: find files by name or content pattern.

Provides grep-like and find-like functionality for exploring codebases.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for files by name pattern or content. Supports glob patterns for filenames and regex for content search. Returns matching file paths with optional line numbers and context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in. Use '.' for current directory.",
                    },
                    "name_pattern": {
                        "type": "string",
                        "description": "Glob pattern for filenames (e.g., '*.py', 'test_*.py'). Optional if content_pattern is provided.",
                    },
                    "content_pattern": {
                        "type": "string",
                        "description": "Text pattern to search for in file contents. Optional if name_pattern is provided.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively in subdirectories. Default: true.",
                        "default": True,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default: 50.",
                        "default": 50,
                    },
                },
                "required": ["path"],
            },
        },
    }


def tool_function(
    path: str,
    name_pattern: str | None = None,
    content_pattern: str | None = None,
    recursive: bool = True,
    max_results: int = 50,
) -> dict[str, Any]:
    """Search for files matching the given criteria.

    Args:
        path: Directory to search in
        name_pattern: Glob pattern for filenames (e.g., "*.py")
        content_pattern: Text to search for in file contents
        recursive: Whether to search subdirectories
        max_results: Maximum results to return

    Returns:
        Dict with search results or error info
    """
    if not name_pattern and not content_pattern:
        return {
            "error": "Must provide either name_pattern or content_pattern (or both)",
            "matches": [],
        }

    search_path = Path(path).expanduser().resolve()
    if not search_path.exists():
        return {"error": f"Path does not exist: {path}", "matches": []}
    if not search_path.is_dir():
        return {"error": f"Path is not a directory: {path}", "matches": []}

    matches = []

    try:
        # Build file list
        if recursive:
            files = list(search_path.rglob("*"))
        else:
            files = list(search_path.glob("*"))

        # Filter to files only
        files = [f for f in files if f.is_file()]

        for file_path in files:
            # Check name pattern
            if name_pattern:
                if not fnmatch.fnmatch(file_path.name, name_pattern):
                    continue

            # Check content pattern
            if content_pattern:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if content_pattern not in content:
                            continue
                except Exception:
                    continue

            matches.append(str(file_path.relative_to(search_path)))

            if len(matches) >= max_results:
                break

        return {
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) >= max_results,
            "search_path": str(search_path),
        }

    except Exception as e:
        return {"error": str(e), "matches": []}
