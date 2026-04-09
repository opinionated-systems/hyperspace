"""
Search tool: find files by name or content.

Provides file search capabilities to locate files within the codebase.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "search_files",
        "description": "Search for files by name pattern or content pattern. Useful for finding files in the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the directory to search in",
                },
                "name_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to match file names (e.g., '*.py', 'test_*.py')",
                },
                "content_pattern": {
                    "type": "string",
                    "description": "Optional regex pattern to search within file contents",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20)",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(
    path: str,
    name_pattern: str | None = None,
    content_pattern: str | None = None,
    max_results: int = 20,
) -> str:
    """Search for files by name or content."""
    try:
        root = Path(path)
        if not root.exists():
            return f"Error: Path does not exist: {path}"
        if not root.is_dir():
            return f"Error: Path is not a directory: {path}"

        results = []
        count = 0

        # Walk through directory
        for item in root.rglob(name_pattern or "*"):
            if count >= max_results:
                break

            if item.is_file():
                match = True

                # Check content pattern if provided
                if content_pattern and match:
                    try:
                        content = item.read_text(encoding="utf-8", errors="ignore")
                        if not re.search(content_pattern, content):
                            match = False
                    except Exception:
                        match = False

                if match:
                    results.append(str(item.absolute()))
                    count += 1

        if not results:
            return "No matching files found."

        return "\n".join(results)
    except Exception as e:
        return f"Error searching files: {e}"
