"""
Search tool: search for patterns in files.

Provides grep-like functionality to search file contents.
"""

from __future__ import annotations

import re
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for a pattern in files. Returns matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path to search in",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If True, search recursively in directories",
                        "default": False,
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    }


def tool_function(pattern: str, path: str, recursive: bool = False) -> str:
    """Search for pattern in file(s).

    Args:
        pattern: Regex pattern to search for
        path: File or directory path to search in
        recursive: If True, search recursively in directories

    Returns:
        String with matching lines and line numbers
    """
    target = Path(path)
    results = []

    if not target.exists():
        return f"Error: Path '{path}' does not exist"

    files_to_search = []
    if target.is_file():
        files_to_search.append(target)
    elif target.is_dir():
        if recursive:
            files_to_search.extend(target.rglob("*.py"))
        else:
            files_to_search.extend(target.glob("*.py"))

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    for file_path in files_to_search:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    results.append(f"{file_path}:{i}: {line.rstrip()}")
        except (IOError, UnicodeDecodeError) as e:
            results.append(f"{file_path}: Error reading file: {e}")

    if not results:
        return f"No matches found for pattern '{pattern}' in '{path}'"

    return "\n".join(results)
