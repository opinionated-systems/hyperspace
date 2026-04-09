"""
File search tool: find files by name pattern or content pattern.

Provides grep-like functionality to search for files and content within the codebase.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for files by name pattern or content pattern. Returns matching file paths with optional content snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in. Defaults to current directory.",
                    },
                    "name_pattern": {
                        "type": "string",
                        "description": "Glob pattern for file names (e.g., '*.py', 'test_*.py'). Optional.",
                    },
                    "content_pattern": {
                        "type": "string",
                        "description": "Text pattern to search for within files. Optional.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default 20.",
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
    max_results: int = 20,
) -> str:
    """Search for files matching the given criteria.

    Args:
        path: Directory path to search in
        name_pattern: Glob pattern for file names (optional)
        content_pattern: Text to search for within files (optional)
        max_results: Maximum number of results to return

    Returns:
        String with matching file paths, one per line
    """
    search_path = Path(path).expanduser().resolve()
    if not search_path.exists():
        return f"Error: Path '{path}' does not exist"
    if not search_path.is_dir():
        return f"Error: Path '{path}' is not a directory"

    matches = []
    count = 0

    for root, dirs, files in os.walk(search_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for filename in files:
            if filename.startswith("."):
                continue

            filepath = Path(root) / filename

            # Check name pattern
            if name_pattern and not fnmatch.fnmatch(filename, name_pattern):
                continue

            # Check content pattern
            if content_pattern:
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if content_pattern not in content:
                            continue
                except (IOError, OSError):
                    continue

            matches.append(str(filepath.relative_to(search_path)))
            count += 1

            if count >= max_results:
                break

        if count >= max_results:
            break

    if not matches:
        return "No matching files found."

    return "\n".join(matches)
