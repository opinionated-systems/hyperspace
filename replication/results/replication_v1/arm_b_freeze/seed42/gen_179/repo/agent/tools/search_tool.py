"""
Search tool: grep-like file content search.

Provides functionality to search for patterns in files within a directory.
"""

from __future__ import annotations

import os
import re
from typing import Any


def _search_in_file(filepath: str, pattern: str, case_sensitive: bool = True) -> list[dict]:
    """Search for pattern in a single file. Returns list of match dicts."""
    matches = []
    flags = 0 if case_sensitive else re.IGNORECASE
    
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                if re.search(pattern, line, flags):
                    matches.append({
                        "line": line_num,
                        "content": line.rstrip("\n"),
                    })
    except (IOError, OSError, UnicodeDecodeError):
        # Skip files we can't read
        pass
    
    return matches


def tool_info() -> dict[str, Any]:
    return {
        "name": "search",
        "description": (
            "Search for a pattern in files within a directory. "
            "Returns file paths and line numbers where matches are found. "
            "Supports regex patterns. Useful for finding code, definitions, or references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: current directory)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Filter by file extension, e.g., '.py' (default: all files)",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default: true)",
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
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for pattern in files."""
    if not pattern:
        return "Error: pattern is required"
    
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a valid directory"
    
    results = []
    total_matches = 0
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for filename in files:
            if filename.startswith("."):
                continue
            
            if file_extension and not filename.endswith(file_extension):
                continue
            
            filepath = os.path.join(root, filename)
            matches = _search_in_file(filepath, pattern, case_sensitive)
            
            if matches:
                rel_path = os.path.relpath(filepath, path)
                for match in matches:
                    if total_matches >= max_results:
                        break
                    results.append(f"{rel_path}:{match['line']}: {match['content']}")
                    total_matches += 1
                
                if total_matches >= max_results:
                    break
        
        if total_matches >= max_results:
            break
    
    if not results:
        return f"No matches found for pattern '{pattern}'"
    
    output = f"Found {total_matches} match(es) for pattern '{pattern}':\n" + "\n".join(results)
    if total_matches >= max_results:
        output += f"\n... (results truncated at {max_results} matches)"
    
    return output
