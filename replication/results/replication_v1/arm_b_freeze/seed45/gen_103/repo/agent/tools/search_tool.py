"""
Search tool: search for patterns in files and directories.

Provides grep-like functionality to find text patterns across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files and directories. "
            "Supports regex patterns and can search specific file types. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search (defaults to allowed root)
        file_extension: Optional extension filter (e.g., '.py')
        max_results: Maximum results to return
    
    Returns:
        Formatted search results with file paths and line numbers
    """
    # Validate pattern
    if not pattern or not pattern.strip():
        return "Error: Empty pattern provided"
    
    # Determine search path
    if path is None:
        if _ALLOWED_ROOT is None:
            return "Error: No path specified and no allowed root set"
        search_path = _ALLOWED_ROOT
    else:
        search_path = os.path.abspath(path)
    
    # Scope check
    if _ALLOWED_ROOT is not None:
        if not search_path.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}. Attempted path: {search_path}"
    
    # Validate search path exists
    p = Path(search_path)
    if not p.exists():
        return f"Error: Path does not exist: {search_path}"
    
    results = []
    count = 0
    
    try:
        if p.is_file():
            # Search single file
            files_to_search = [p]
        else:
            # Search directory
            if file_extension:
                files_to_search = list(p.rglob(f"*{file_extension}"))
            else:
                files_to_search = list(p.rglob("*"))
            # Filter to files only and exclude hidden directories
            files_to_search = [
                f for f in files_to_search 
                if f.is_file() 
                and not any(part.startswith('.') for part in f.relative_to(p).parts)
            ]
        
        # Compile regex
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        
        # Search files
        for file_path in files_to_search:
            if count >= max_results:
                break
            
            # Skip binary files and very large files
            try:
                size = file_path.stat().st_size
                if size > 10_000_000:  # Skip files > 10MB
                    continue
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if count >= max_results:
                            break
                        
                        if regex.search(line):
                            # Truncate long lines
                            if len(line) > 200:
                                line = line[:100] + "..." + line[-100:]
                            results.append(f"{file_path}:{line_num}: {line.rstrip()}")
                            count += 1
            except (IOError, OSError, UnicodeDecodeError):
                # Skip files that can't be read
                continue
    
    except Exception as e:
        return f"Error during search: {e}"
    
    if not results:
        return f"No matches found for pattern '{pattern}' in {search_path}"
    
    header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
    return header + "\n".join(results)
