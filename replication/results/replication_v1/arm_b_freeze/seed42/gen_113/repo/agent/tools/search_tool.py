"""
Search tool for finding files by content pattern.

Provides grep-like functionality to search within files.
"""

from __future__ import annotations

import os
import re
import fnmatch
from typing import Iterator


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "search_files",
        "description": (
            "Search for files containing a regex pattern. "
            "Returns file paths with line numbers and matching lines. "
            "Useful for finding code patterns across the codebase. "
            "Supports case-insensitive search, file glob patterns, and depth limits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search recursively",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for in file contents",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py')",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern for filenames (e.g., '*.py', 'test_*.py')",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive (default: true)",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth to search (default: unlimited)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show around matches (default: 0)",
                },
            },
            "required": ["path", "pattern"],
        },
    }


def _walk_with_depth(path: str, max_depth: int | None = None) -> Iterator[tuple[str, list[str], list[str]]]:
    """Walk directory tree with optional depth limit.
    
    Yields same tuples as os.walk: (root, dirs, files)
    """
    base_depth = path.rstrip(os.sep).count(os.sep)
    
    for root, dirs, files in os.walk(path):
        if max_depth is not None:
            current_depth = root.count(os.sep) - base_depth
            if current_depth >= max_depth:
                # Don't descend further
                dirs[:] = []
        yield root, dirs, files


def tool_function(
    path: str, 
    pattern: str, 
    file_extension: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = True,
    max_depth: int | None = None,
    context_lines: int = 0,
) -> str:
    """Search for files containing a pattern.

    Args:
        path: Directory path to search recursively
        pattern: Regex pattern to search for
        file_extension: Optional file extension filter (e.g., '.py')
        file_pattern: Optional glob pattern for filenames (e.g., '*.py')
        case_sensitive: Whether search is case-sensitive (default: true)
        max_depth: Maximum directory depth to search (default: unlimited)
        context_lines: Number of context lines around matches (default: 0)

    Returns:
        String with matching files and line numbers
    """
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a valid directory"

    matches = []
    lines_buffer = []  # For context lines
    
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    # Determine file filter function
    def file_matches(filename: str) -> bool:
        if file_extension and not filename.endswith(file_extension):
            return False
        if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
            return False
        return True

    for root, _, files in _walk_with_depth(path, max_depth):
        for filename in files:
            if not file_matches(filename):
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = list(f)
                    
                    for line_num, line in enumerate(lines, 1):
                        if compiled_pattern.search(line):
                            # Build match with context
                            match_lines = []
                            
                            # Add context before
                            start = max(0, line_num - context_lines - 1)
                            for i in range(start, line_num - 1):
                                match_lines.append(f"  {i+1}: {lines[i].rstrip()}")
                            
                            # Add the matching line
                            match_lines.append(f"> {line_num}: {line.rstrip()}")
                            
                            # Add context after
                            end = min(len(lines), line_num + context_lines)
                            for i in range(line_num, end):
                                match_lines.append(f"  {i+1}: {lines[i].rstrip()}")
                            
                            match_text = f"{filepath}:\n" + "\n".join(match_lines)
                            matches.append(match_text)
                            
                            if len(matches) >= 100:  # Limit results
                                break
                    
                    if len(matches) >= 100:
                        break
            except Exception:
                continue
        if len(matches) >= 100:
            break

    if not matches:
        return "No matches found."

    result = "\n\n".join(matches[:30])  # Return first 30 matches
    if len(matches) > 30:
        result += f"\n\n... ({len(matches) - 30} more matches)"
    return result
