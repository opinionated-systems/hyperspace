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
                    "description": "Absolute path to file or directory to search in. Defaults to current directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in directories. Default: True.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default: False.",
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
    recursive: bool = True,
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.
    
    Returns matching lines with file paths and line numbers.
    """
    if not pattern:
        return "Error: pattern is required."
    
    # Default to current directory if no path provided
    if path is None:
        path = os.getcwd()
    
    p = Path(path)
    
    if not p.is_absolute():
        return f"Error: {path} is not an absolute path."
    
    # Scope check: only allow operations within the allowed root
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    results = []
    max_results = 100  # Limit to prevent overwhelming output
    
    try:
        if p.is_file():
            # Search single file
            files_to_search = [p]
        elif p.is_dir():
            # Search directory
            if recursive:
                files_to_search = list(p.rglob("*"))
            else:
                files_to_search = list(p.iterdir())
            # Filter to files only
            files_to_search = [f for f in files_to_search if f.is_file()]
        else:
            return f"Error: {p} does not exist."
        
        # Filter by extension if specified
        if file_extension:
            files_to_search = [f for f in files_to_search if f.suffix == file_extension]
        
        # Skip binary files and common non-text files
        skip_extensions = {'.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', 
                         '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
                         '.pdf', '.zip', '.tar', '.gz', '.bz2', '.7z',
                         '.db', '.sqlite', '.sqlite3'}
        
        for file_path in files_to_search:
            if file_path.suffix in skip_extensions:
                continue
            
            try:
                # Skip files that are too large (>1MB)
                if file_path.stat().st_size > 1_000_000:
                    continue
                
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    if compiled_pattern.search(line):
                        # Truncate very long lines
                        if len(line) > 200:
                            line = line[:100] + " ... " + line[-100:]
                        results.append(f"{file_path}:{line_num}: {line}")
                        
                        if len(results) >= max_results:
                            break
                
                if len(results) >= max_results:
                    break
                    
            except (IOError, OSError, UnicodeDecodeError):
                # Skip files we can't read
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}' in {p}"
        
        result_text = '\n'.join(results)
        if len(results) >= max_results:
            result_text += f"\n\n[Search truncated at {max_results} matches]"
        
        return f"Found {len(results)} matches for pattern '{pattern}':\n{result_text}"
        
    except Exception as e:
        return f"Error during search: {type(e).__name__}: {e}"
