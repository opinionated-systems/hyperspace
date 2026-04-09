"""
Search tool: grep-like functionality to find patterns in files.

Provides file search capabilities to locate code patterns across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata matching the paper's tool interface."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep-like functionality. Can search within specific files or recursively in directories.",
        "parameters": {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory path to search in. Defaults to current directory.",
                "default": ".",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to search recursively in directories. Defaults to True.",
                "default": True,
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g., '*.py'). Defaults to all files.",
                "default": None,
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether the search is case sensitive. Defaults to True.",
                "default": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 50.",
                "default": 50,
            },
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    file_pattern: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for patterns in files.
    
    Args:
        pattern: The regex pattern to search for
        path: File or directory path to search in
        recursive: Whether to search recursively in directories
        file_pattern: Glob pattern to filter files (e.g., '*.py')
        case_sensitive: Whether the search is case sensitive
        max_results: Maximum number of results to return
    
    Returns:
        Search results as a formatted string
    """
    try:
        search_path = Path(path).expanduser().resolve()
        
        if not search_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        if search_path.is_file():
            # Search in single file
            files_to_search = [search_path]
        else:
            # Search in directory
            if recursive:
                if file_pattern:
                    files_to_search = list(search_path.rglob(file_pattern))
                else:
                    files_to_search = list(search_path.rglob("*"))
            else:
                if file_pattern:
                    files_to_search = list(search_path.glob(file_pattern))
                else:
                    files_to_search = list(search_path.glob("*"))
            
            # Filter to only files
            files_to_search = [f for f in files_to_search if f.is_file()]
        
        # Search in files
        for file_path in files_to_search:
            # Skip binary files and very large files
            try:
                if file_path.stat().st_size > 10 * 1024 * 1024:  # Skip files > 10MB
                    continue
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Format: file_path:line_num: line_content
                            rel_path = file_path.relative_to(Path.cwd()) if file_path.is_relative_to(Path.cwd()) else file_path
                            results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                            
                            if len(results) >= max_results:
                                break
                    
                    if len(results) >= max_results:
                        break
            except (IOError, OSError, UnicodeDecodeError):
                # Skip files that can't be read
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}' in '{path}'"
        
        header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
        if len(results) >= max_results:
            header = f"Found {len(results)}+ match(es) for pattern '{pattern}' (showing first {max_results}):\n"
        
        return header + "\n".join(results)
    
    except re.error as e:
        return f"Error: Invalid regex pattern - {e}"
    except Exception as e:
        return f"Error during search: {e}"


if __name__ == "__main__":
    # Simple test
    print(tool_function("def ", path=".", file_pattern="*.py", max_results=10))
