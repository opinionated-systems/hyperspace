"""
Search tool: search for patterns in files using grep-like functionality.

Provides file search capabilities to help agents find code patterns,
function definitions, and references across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for the search tool."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep-like functionality. Can search for text patterns, regex patterns, or file names. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (text or regex)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in. Defaults to current directory.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', '*.js'). Defaults to all files.",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Whether the pattern is a regex. Defaults to False (literal search).",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Defaults to True.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 50.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    is_regex: bool = False,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for patterns in files.
    
    Args:
        pattern: The search pattern (text or regex)
        path: Directory or file path to search in
        file_pattern: File pattern to match (e.g., '*.py')
        is_regex: Whether the pattern is a regex
        case_sensitive: Whether the search is case sensitive
        max_results: Maximum number of results to return
    
    Returns:
        String with search results, including file paths and line numbers
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    
    results = []
    count = 0
    
    # Compile regex pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    if is_regex:
        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
    else:
        compiled_pattern = re.compile(re.escape(pattern), flags)
    
    # Walk through directory
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            # Skip hidden files
            if filename.startswith('.'):
                continue
            
            # Check file pattern
            if file_pattern and not _match_pattern(filename, file_pattern):
                continue
            
            filepath = os.path.join(root, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled_pattern.search(line):
                            # Truncate long lines
                            display_line = line.rstrip()
                            if len(display_line) > 200:
                                display_line = display_line[:200] + "..."
                            
                            rel_path = os.path.relpath(filepath, path) if path != "." else filepath
                            results.append(f"{rel_path}:{line_num}: {display_line}")
                            count += 1
                            
                            if count >= max_results:
                                results.append(f"\n... (truncated, showing first {max_results} matches)")
                                return "\n".join(results)
            except (IOError, OSError, UnicodeDecodeError):
                # Skip files that can't be read
                continue
    
    if not results:
        return f"No matches found for pattern '{pattern}'"
    
    return f"Found {count} match(es):\n" + "\n".join(results)


def _match_pattern(filename: str, pattern: str) -> bool:
    """Match filename against a glob-like pattern."""
    # Simple glob matching
    import fnmatch
    return fnmatch.fnmatch(filename, pattern)
