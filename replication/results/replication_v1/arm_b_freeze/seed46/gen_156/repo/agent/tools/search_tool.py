"""
Search tool: search for patterns in files using grep-like functionality.

Provides structured file search capabilities for the agentic loop.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep. Searches file contents for matching lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in. Default is current directory.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt'). Searches all files if not specified.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default is True.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return. Default is 50.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str = "",
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for pattern in files.

    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional extension filter (e.g., '.py')
        case_sensitive: Whether search is case sensitive
        max_results: Maximum matches to return

    Returns:
        Search results as formatted string
    """
    if not pattern:
        return "Error: pattern is required"

    # Validate path exists
    if not os.path.exists(path):
        return f"Error: path '{path}' does not exist"

    results = []
    count = 0
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    # Collect files to search
    files_to_search = []
    if os.path.isfile(path):
        files_to_search.append(path)
    else:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for filename in files:
                if filename.startswith('.'):
                    continue
                if file_extension and not filename.endswith(file_extension):
                    continue
                files_to_search.append(os.path.join(root, filename))

    # Search files
    for filepath in files_to_search:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if compiled_pattern.search(line):
                        # Truncate long lines
                        display_line = line.rstrip()
                        if len(display_line) > 200:
                            display_line = display_line[:100] + " ... " + display_line[-100:]
                        results.append(f"{filepath}:{line_num}:{display_line}")
                        count += 1
                        if count >= max_results:
                            break
        except (IOError, OSError, UnicodeDecodeError):
            # Skip files that can't be read
            continue

        if count >= max_results:
            break

    if not results:
        return f"No matches found for pattern '{pattern}'"

    header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
    return header + "\n".join(results)
