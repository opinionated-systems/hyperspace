"""
Search tool: search for patterns in files.

Provides grep-like functionality to search file contents.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict:
    """Return tool specification for OpenAI function calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for a pattern in files within a directory. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (recursive)",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., '.py', '.js')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50)",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    }


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """Search for pattern in files.
    
    Args:
        pattern: Regular expression to search for
        path: Directory to search in
        file_extension: Optional extension filter
        max_results: Maximum results to return
        
    Returns:
        Dict with search results or error info
    """
    if not os.path.isdir(path):
        return {
            "success": False,
            "error": f"Path is not a directory: {path}",
        }
    
    results = []
    count = 0
    
    try:
        compiled_pattern = re.compile(pattern)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
        }
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            if filename.startswith('.'):
                continue
            if file_extension and not filename.endswith(file_extension):
                continue
                
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled_pattern.search(line):
                            results.append({
                                "file": filepath,
                                "line": line_num,
                                "content": line.rstrip('\n'),
                            })
                            count += 1
                            if count >= max_results:
                                break
                    if count >= max_results:
                        break
            except (IOError, OSError, UnicodeDecodeError):
                continue
                
        if count >= max_results:
            break
    
    return {
        "success": True,
        "pattern": pattern,
        "path": path,
        "matches_found": len(results),
        "truncated": count >= max_results,
        "results": results,
    }
