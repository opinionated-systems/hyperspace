"""
Search tool: search for patterns in files using grep-like functionality.

Provides file search capabilities to complement bash and editor tools.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for the search tool."""
    return {
        "name": "search",
        "description": "Search for patterns in files within a directory. Supports regex patterns and can limit search depth.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: current directory)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt')",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth to search (default: unlimited)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    max_depth: int | None = None,
    max_results: int = 50,
) -> str:
    """Search for a regex pattern in files.

    Args:
        pattern: Regex pattern to search for
        path: Directory path to search in
        file_extension: Optional file extension filter
        max_depth: Maximum directory depth to search
        max_results: Maximum number of results to return

    Returns:
        String with search results or error message
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a directory"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results = []
    result_count = 0

    for root, dirs, files in os.walk(path):
        # Check depth limit
        if max_depth is not None:
            current_depth = root.count(os.sep) - path.count(os.sep)
            if current_depth >= max_depth:
                del dirs[:]
                continue

        for filename in files:
            # Filter by extension if specified
            if file_extension and not filename.endswith(file_extension):
                continue

            filepath = os.path.join(root, filename)

            # Skip binary files and very large files
            try:
                if os.path.getsize(filepath) > 1024 * 1024:  # Skip files > 1MB
                    continue
            except OSError:
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Show context around match
                            match_str = line.strip()
                            if len(match_str) > 100:
                                match_str = match_str[:97] + "..."
                            results.append(f"{filepath}:{line_num}: {match_str}")
                            result_count += 1

                            if result_count >= max_results:
                                results.append(f"\n[Results truncated after {max_results} matches]")
                                return "\n".join(results)
            except (IOError, OSError, UnicodeDecodeError):
                continue

    if not results:
        return f"No matches found for pattern '{pattern}' in '{path}'"

    header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
    return header + "\n".join(results)
