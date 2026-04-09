"""
Search tool: search for files by name or content.

Provides grep-like and find-like functionality for exploring codebases.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata in OpenAI function format."""
    return {
        "name": "search",
        "description": "Search for files by name pattern or content pattern. Equivalent to grep -r and find commands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: current directory)",
                },
                "content_pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for in file contents (optional)",
                },
                "name_pattern": {
                    "type": "string",
                    "description": "Glob pattern to match file names (e.g., '*.py', optional)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                },
            },
            "required": [],
        },
    }


def tool_function(
    path: str = ".",
    content_pattern: str | None = None,
    name_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for files matching the given criteria.

    Args:
        path: Directory to search in
        content_pattern: Regex pattern to search in file contents
        name_pattern: Glob pattern for file names (e.g., "*.py")
        max_results: Maximum number of results to return

    Returns:
        String with search results
    """
    results = []
    count = 0

    # Convert name_pattern to regex if provided
    name_regex = None
    if name_pattern:
        # Convert glob to regex
        name_regex = name_pattern.replace(".", r"\.")
        name_regex = name_regex.replace("*", ".*")
        name_regex = name_regex.replace("?", ".")
        name_regex = re.compile(name_regex + "$", re.IGNORECASE)

    # Compile content pattern if provided
    content_regex = None
    if content_pattern:
        try:
            content_regex = re.compile(content_pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid content pattern: {e}"

    try:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for filename in files:
                # Check name pattern
                if name_regex and not name_regex.match(filename):
                    continue

                filepath = os.path.join(root, filename)

                # Skip binary files and hidden files
                if filename.startswith("."):
                    continue

                # Check content pattern
                if content_regex:
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            matches = list(content_regex.finditer(content))
                            if matches:
                                for match in matches[:3]:  # Show up to 3 matches per file
                                    line_num = content[:match.start()].count("\n") + 1
                                    line_content = content.split("\n")[line_num - 1].strip()
                                    results.append(f"{filepath}:{line_num}: {line_content[:100]}")
                                    count += 1
                                    if count >= max_results:
                                        break
                            else:
                                continue  # No content match, skip this file
                    except (IOError, OSError, UnicodeDecodeError):
                        continue  # Skip files we can't read
                else:
                    # No content pattern, just list the file
                    results.append(filepath)
                    count += 1

                if count >= max_results:
                    break

            if count >= max_results:
                break

    except Exception as e:
        return f"Error during search: {e}"

    if not results:
        return "No matches found."

    if count >= max_results:
        results.append(f"\n... (showing first {max_results} results)")

    return "\n".join(results)
