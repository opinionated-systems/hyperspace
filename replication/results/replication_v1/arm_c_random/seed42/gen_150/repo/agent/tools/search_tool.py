"""
Search tool: grep-like text search for exploring codebases.

Provides content-based search to find files containing specific patterns.
Complements file_tool and editor_tool for codebase exploration.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for search operations."""
    return {
        "name": "search",
        "description": "Search for text patterns in files using grep-like functionality. Searches file contents for matches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text pattern to search for (supports basic regex)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py', '*.js')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20)",
                },
            },
            "required": ["pattern", "path"],
        },
    }


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    max_results: int = 20,
) -> str:
    """Search for pattern in files.

    Args:
        pattern: Text pattern to search for
        path: Directory or file path to search in
        file_pattern: Optional glob pattern to filter files
        max_results: Maximum number of results to return

    Returns:
        Search results as formatted string
    """
    p = Path(path)
    results = []
    count = 0

    if not p.exists():
        return f"Error: path '{path}' does not exist"

    # Determine files to search
    if p.is_file():
        files = [p]
    else:
        if file_pattern:
            files = list(p.rglob(file_pattern))
        else:
            files = [f for f in p.rglob("*") if f.is_file()]

    # Search in files
    for file_path in files:
        if count >= max_results:
            break

        # Skip binary files and very large files
        try:
            if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                continue
        except (OSError, IOError):
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if count >= max_results:
                        break
                    try:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Truncate long lines
                            line_display = line.rstrip()
                            if len(line_display) > 200:
                                line_display = line_display[:197] + "..."
                            rel_path = file_path.relative_to(p) if p.is_dir() else file_path.name
                            results.append(f"{rel_path}:{line_num}: {line_display}")
                            count += 1
                    except re.error:
                        # If regex fails, try literal string match
                        if pattern.lower() in line.lower():
                            line_display = line.rstrip()
                            if len(line_display) > 200:
                                line_display = line_display[:197] + "..."
                            rel_path = file_path.relative_to(p) if p.is_dir() else file_path.name
                            results.append(f"{rel_path}:{line_num}: {line_display}")
                            count += 1
        except (OSError, IOError, UnicodeDecodeError):
            # Skip files that can't be read
            continue

    if not results:
        return f"No matches found for pattern '{pattern}' in '{path}'"

    header = f"Found {len(results)} match(es) for '{pattern}':\n"
    return header + "\n".join(results)
