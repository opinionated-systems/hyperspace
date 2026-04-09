"""
Search tool for finding text patterns in files.

Provides grep-like functionality to search for patterns within files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata matching the expected schema."""
    return {
        "name": "search",
        "description": "Search for text patterns in files. Supports regex patterns and can search recursively in directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to search in. If directory, searches recursively.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py', '*.md'). Only used when path is a directory.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive. Default is False.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default is 50.",
                },
            },
            "required": ["pattern", "path"],
        },
    }


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The regex pattern to search for
        path: File or directory path to search in
        file_pattern: Optional glob pattern to filter files
        case_sensitive: Whether search is case sensitive
        max_results: Maximum number of results to return
    
    Returns:
        String with search results or error message
    """
    try:
        target_path = Path(path).expanduser().resolve()
        
        if not target_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled_pattern = re.compile(pattern, flags)
        
        results = []
        
        if target_path.is_file():
            # Search single file
            files_to_search = [target_path]
        else:
            # Search directory recursively
            if file_pattern:
                files_to_search = list(target_path.rglob(file_pattern))
            else:
                files_to_search = [
                    f for f in target_path.rglob("*") 
                    if f.is_file() and not any(part.startswith(".") for part in f.relative_to(target_path).parts)
                ]
        
        for file_path in files_to_search:
            if not file_path.is_file():
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    if compiled_pattern.search(line):
                        # Show context: line number and content
                        result_line = f"{file_path}:{line_num}:{line.rstrip()}"
                        results.append(result_line)
                        
                        if len(results) >= max_results:
                            break
                            
                if len(results) >= max_results:
                    break
                    
            except (IOError, OSError, PermissionError) as e:
                results.append(f"{file_path}: Error reading file: {e}")
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}' in '{path}'"
        
        output = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
        output += "\n".join(results)
        
        if len(results) >= max_results:
            output += f"\n... (results truncated at {max_results} matches)"
        
        return output
        
    except re.error as e:
        return f"Error: Invalid regex pattern '{pattern}': {e}"
    except Exception as e:
        return f"Error searching: {e}"
