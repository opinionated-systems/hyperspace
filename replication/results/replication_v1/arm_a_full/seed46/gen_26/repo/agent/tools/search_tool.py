"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns
within files in the repository.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict:
    return {
        "name": "search",
        "description": "Search for a pattern in files within a directory. Returns matching lines with file paths and line numbers. Supports regex patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (recursive)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt'). Empty string searches all files.",
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
    file_extension: str = "",
    max_results: int = 50,
) -> str:
    """Search for pattern in files and return matching lines."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a directory"
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    matches = []
    match_count = 0
    files_searched = 0
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            if filename.startswith('.'):
                continue
            
            if file_extension and not filename.endswith(file_extension):
                continue
            
            filepath = os.path.join(root, filename)
            files_searched += 1
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Show context: file:line_num: content
                            matches.append(f"{filepath}:{line_num}: {line.rstrip()}")
                            match_count += 1
                            
                            if match_count >= max_results:
                                break
                    
                    if match_count >= max_results:
                        break
            except (IOError, OSError, UnicodeDecodeError):
                # Skip files we can't read
                continue
        
        if match_count >= max_results:
            break
    
    if not matches:
        return f"No matches found for pattern '{pattern}' in {files_searched} files searched."
    
    result = f"Found {match_count} match(es) in {files_searched} files searched:\n"
    result += "\n".join(matches)
    
    if match_count >= max_results:
        result += f"\n... (truncated at {max_results} matches)"
    
    return result
