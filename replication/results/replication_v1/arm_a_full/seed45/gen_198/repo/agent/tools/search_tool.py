"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns within files.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    return {
        "name": "search",
        "description": "Search for patterns in files using regex or literal text matching. Returns matching lines with context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The pattern to search for (regex or literal text)",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to search in. If directory, searches all .py files recursively.",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Whether to treat pattern as regex (default: false for literal search)",
                    "default": False,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive (default: true)",
                    "default": True,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before and after matches (default: 2)",
                    "default": 2,
                },
            },
            "required": ["pattern", "path"],
        },
    }


def tool_function(
    pattern: str,
    path: str,
    is_regex: bool = False,
    case_sensitive: bool = True,
    context_lines: int = 2,
) -> str:
    """Search for pattern in file(s) and return matches with context."""
    if not pattern:
        return "Error: pattern is required"
    
    if not path:
        return "Error: path is required"
    
    # Expand user home directory
    path = os.path.expanduser(path)
    
    # Check if path exists
    if not os.path.exists(path):
        return f"Error: path '{path}' does not exist"
    
    # Prepare regex pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        if is_regex:
            regex = re.compile(pattern, flags)
        else:
            # Escape literal text for regex
            regex = re.compile(re.escape(pattern), flags)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"
    
    results = []
    
    def search_file(filepath: str) -> list[str]:
        """Search a single file and return formatted matches."""
        file_results = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            return [f"  Error reading {filepath}: {e}"]
        
        matches_found = False
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                if not matches_found:
                    file_results.append(f"\n{filepath}:")
                    matches_found = True
                
                # Show context lines
                start = max(0, i - context_lines - 1)
                end = min(len(lines), i + context_lines)
                
                for j in range(start, end):
                    line_num = j + 1
                    prefix = ">>> " if j == i - 1 else "    "
                    content = lines[j].rstrip('\n')
                    # Truncate very long lines
                    if len(content) > 200:
                        content = content[:100] + " ... [truncated] ... " + content[-100:]
                    file_results.append(f"{prefix}{line_num:4d}: {content}")
                
                file_results.append("")  # Blank line between match groups
        
        return file_results
    
    if os.path.isfile(path):
        # Search single file
        results.extend(search_file(path))
    else:
        # Search directory recursively for .py files
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for filename in files:
                if filename.endswith('.py'):
                    filepath = os.path.join(root, filename)
                    results.extend(search_file(filepath))
    
    if not results:
        return f"No matches found for pattern '{pattern}' in '{path}'"
    
    # Summary
    summary = f"Search results for pattern '{pattern}' in '{path}':"
    return summary + "\n".join(results)
