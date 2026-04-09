"""
File search tool: search for files by name or content pattern.

Provides capabilities to:
- Search for files by name pattern (glob)
- Search for files containing specific text content
- Search recursively within a directory
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
            "description": "Search for files by name pattern or content. Can search recursively within a directory for files matching a name pattern (glob) or containing specific text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in. Must be absolute.",
                    },
                    "name_pattern": {
                        "type": "string",
                        "description": "Optional glob pattern for file names (e.g., '*.py', 'test_*.py'). If not provided, searches all files.",
                    },
                    "content_pattern": {
                        "type": "string",
                        "description": "Optional text to search for within file contents. If provided, only files containing this text are returned.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 50).",
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
    max_results: int = 50,
) -> str:
    """Search for files matching criteria.

    Args:
        path: Directory path to search (absolute)
        name_pattern: Optional glob pattern for file names
        content_pattern: Optional text to search in file contents
        max_results: Maximum results to return

    Returns:
        String with search results or error message
    """
    # Validate path
    if not os.path.isabs(path):
        return f"Error: Path must be absolute, got: {path}"

    if not os.path.isdir(path):
        return f"Error: Not a valid directory: {path}"

    results = []
    count = 0

    try:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for filename in files:
                if count >= max_results:
                    break

                # Check name pattern
                if name_pattern and not fnmatch.fnmatch(filename, name_pattern):
                    continue

                filepath = os.path.join(root, filename)

                # Check content pattern
                if content_pattern:
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if content_pattern not in content:
                                continue
                    except (IOError, OSError, UnicodeDecodeError):
                        # Skip files that can't be read
                        continue

                # Add to results
                rel_path = os.path.relpath(filepath, path)
                results.append(rel_path)
                count += 1

            if count >= max_results:
                break

    except Exception as e:
        return f"Error during search: {str(e)}"

    if not results:
        return "No files found matching the criteria."

    output = f"Found {len(results)} file(s):\n"
    output += "\n".join(results)
    if count >= max_results:
        output += f"\n... (limited to {max_results} results)"

    return output
