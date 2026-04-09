"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns within files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for a pattern in files within a directory. Returns matching lines with file paths and line numbers. Useful for finding code patterns, function definitions, or specific text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the directory or file to search in",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt'). If not provided, searches all files.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 50)",
                },
            },
            "required": ["pattern", "path"],
        },
    }


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Path does not exist: {path}"

        matches = []
        count = 0

        # Compile regex pattern
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        # Determine files to search
        if p.is_file():
            files = [p]
        else:
            files = p.rglob("*")

        for file_path in files:
            if not file_path.is_file():
                continue
            if file_extension and not str(file_path).endswith(file_extension):
                continue
            # Skip binary files
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append({
                                "file": str(file_path),
                                "line": line_num,
                                "content": line.rstrip("\n"),
                            })
                            count += 1
                            if count >= max_results:
                                break
                    if count >= max_results:
                        break
            except (IOError, OSError):
                continue

        if not matches:
            return f"No matches found for pattern '{pattern}' in {path}"

        # Format results
        lines = [f"Found {len(matches)} matches for pattern '{pattern}':"]
        for m in matches:
            lines.append(f"{m['file']}:{m['line']}: {m['content']}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error searching: {e}"
