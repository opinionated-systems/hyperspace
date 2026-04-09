"""
Search tool: search for content within files.

Provides grep-like functionality to find patterns in files.
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
            "description": "Search for patterns in files using regex or literal text. Returns matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The search pattern (regex or literal text)",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path to search in",
                    },
                    "is_regex": {
                        "type": "boolean",
                        "description": "Whether pattern is a regex (default: false)",
                        "default": False,
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search is case sensitive (default: true)",
                        "default": True,
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    }


def tool_function(
    pattern: str,
    path: str,
    is_regex: bool = False,
    case_sensitive: bool = True,
) -> str:
    """Search for pattern in file(s).
    
    Args:
        pattern: Search pattern (regex or literal)
        path: File or directory to search
        is_regex: Treat pattern as regex if True
        case_sensitive: Case sensitive search if True
    
    Returns:
        Matching lines with line numbers, or error message
    """
    try:
        target = Path(path)
        
        if not target.exists():
            return f"Error: Path not found: {path}"
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regex:
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"
        else:
            regex = re.compile(re.escape(pattern), flags)
        
        results = []
        
        def search_file(file_path: Path) -> None:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{file_path}:{i}: {line.rstrip()}")
            except Exception:
                pass  # Skip files that can't be read
        
        if target.is_file():
            search_file(target)
        elif target.is_dir():
            for file_path in target.rglob("*"):
                if file_path.is_file():
                    search_file(file_path)
        
        if not results:
            return f"No matches found for '{pattern}' in {path}"
        
        # Limit output to avoid overwhelming responses
        max_results = 50
        if len(results) > max_results:
            return "\n".join(results[:max_results]) + f"\n... ({len(results) - max_results} more matches)"
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error searching: {e}"
