"""
File search tool: search for files by name pattern or content pattern.

Provides recursive file search capabilities within a directory.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "file_search",
        "description": (
            "Search for files by name pattern or content pattern within a directory. "
            "Returns matching file paths with optional preview of content matches. "
            "Use name_pattern for glob matching (e.g., '*.py', 'test_*.json'). "
            "Use content_pattern to search within file contents. "
            "At least one of name_pattern or content_pattern must be provided."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to search in (absolute or relative).",
                },
                "name_pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names (e.g., '*.py', 'test_*.json'). Optional if content_pattern is provided.",
                },
                "content_pattern": {
                    "type": "string",
                    "description": "Text pattern to search for within file contents. Optional if name_pattern is provided.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories. Default: true.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
            },
            "required": ["directory"],
        },
    }


def tool_function(
    directory: str,
    name_pattern: str | None = None,
    content_pattern: str | None = None,
    recursive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for files matching the given criteria.

    Args:
        directory: Directory path to search in.
        name_pattern: Glob pattern for file names.
        content_pattern: Text pattern to search within files.
        recursive: Whether to search subdirectories.
        max_results: Maximum number of results.

    Returns:
        String with matching file paths, one per line.
    """
    if not name_pattern and not content_pattern:
        return "Error: At least one of name_pattern or content_pattern must be provided."

    root = Path(directory).expanduser().resolve()
    if not root.exists():
        return f"Error: Directory '{directory}' does not exist."
    if not root.is_dir():
        return f"Error: Path '{directory}' is not a directory."

    matches = []
    files_searched = 0

    # Determine walk strategy
    if recursive:
        walker = os.walk(root)
    else:
        walker = [(str(root), [], [f.name for f in root.iterdir() if f.is_file()])]

    for dirpath, _dirnames, filenames in walker:
        for filename in filenames:
            # Check name pattern
            if name_pattern and not fnmatch.fnmatch(filename, name_pattern):
                continue

            filepath = Path(dirpath) / filename

            # Check content pattern
            if content_pattern:
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if content_pattern not in content:
                            continue
                except (IOError, OSError, UnicodeDecodeError):
                    continue

            matches.append(str(filepath))
            files_searched += 1

            if len(matches) >= max_results:
                break

        if len(matches) >= max_results:
            break

    if not matches:
        return "No matching files found."

    result = f"Found {len(matches)} file(s):\n"
    result += "\n".join(matches)
    return result
