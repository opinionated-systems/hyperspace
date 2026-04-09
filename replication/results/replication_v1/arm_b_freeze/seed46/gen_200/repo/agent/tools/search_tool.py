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
            "Supports regex patterns and can search recursively. "
            "Returns matching file paths with line numbers and context."
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
                    "description": "Absolute path to file or directory to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default 50.",
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
        file_extension: Optional filter by file extension
        max_results: Maximum number of matches to return
    
    Returns:
        Formatted search results with file paths, line numbers, and context
    """
    # Validate pattern is not empty
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
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    p = Path(search_path)
    
    if not p.exists():
        return f"Error: {search_path} does not exist"
    
    results = []
    count = 0
    
    try:
        if p.is_file():
            # Search single file
            files_to_search = [p]
        else:
            # Search directory recursively
            if file_extension:
                files_to_search = list(p.rglob(f"*{file_extension}"))
            else:
                files_to_search = list(p.rglob("*"))
            # Filter to only files, exclude hidden directories
            files_to_search = [
                f for f in files_to_search 
                if f.is_file() and not any(part.startswith('.') for part in f.parts)
            ]
        
        for file_path in files_to_search:
            if count >= max_results:
                break
            
            # Skip binary files and very large files
            try:
                stat = file_path.stat()
                if stat.st_size > 10 * 1024 * 1024:  # Skip files > 10MB
                    continue
            except OSError:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines, 1):
                    if count >= max_results:
                        break
                    
                    try:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Show context: line content (truncated if too long)
                            content = line.strip()
                            if len(content) > 100:
                                content = content[:97] + "..."
                            rel_path = file_path.relative_to(p) if p.is_dir() else file_path.name
                            results.append(f"{rel_path}:{i}: {content}")
                            count += 1
                    except re.error as e:
                        return f"Error: Invalid regex pattern: {e}"
                        
            except (IOError, OSError, PermissionError):
                continue
    
    except Exception as e:
        return f"Error during search: {e}"
    
    if not results:
        return f"No matches found for pattern '{pattern}' in {search_path}"
    
    header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
    if count >= max_results:
        header = f"Found {max_results}+ match(es) for pattern '{pattern}' (showing first {max_results}):\n"
    
    return header + "\n".join(results)
