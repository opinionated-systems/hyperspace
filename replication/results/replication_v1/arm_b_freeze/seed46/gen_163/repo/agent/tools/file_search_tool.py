"""
File search tool: search for files by name pattern.

Provides a way to find files in the codebase by name pattern,
which is useful for exploring and understanding the codebase structure.
"""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_search",
        "description": (
            "Search for files by name pattern. "
            "Supports wildcards like *.py, test_*.py, etc. "
            "Returns matching file paths relative to the search root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "File name pattern to search for (e.g., '*.py', 'test_*.py').",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path). Defaults to allowed root.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively. Default: true.",
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
) -> str:
    """Search for files matching the given pattern.
    
    Args:
        pattern: File name pattern (e.g., '*.py', 'test_*.py')
        path: Directory to search in (absolute path). Defaults to allowed root.
        recursive: Whether to search recursively. Default: True.
    
    Returns:
        String with matching file paths, one per line.
    """
    # Determine search root
    if path is None:
        if _ALLOWED_ROOT is None:
            return "Error: No search path provided and no allowed root set."
        search_root = _ALLOWED_ROOT
    else:
        search_root = os.path.abspath(path)
    
    # Validate path is absolute
    if not os.path.isabs(search_root):
        return f"Error: {search_root} is not an absolute path."
    
    # Scope check: only allow operations within the allowed root
    if _ALLOWED_ROOT is not None:
        if not search_root.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    # Check if path exists
    if not os.path.exists(search_root):
        return f"Error: Path does not exist: {search_root}"
    
    if not os.path.isdir(search_root):
        return f"Error: Path is not a directory: {search_root}"
    
    # Perform search
    matches = []
    try:
        if recursive:
            for root_dir, dirs, files in os.walk(search_root):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for filename in files:
                    if not filename.startswith('.') and fnmatch.fnmatch(filename, pattern):
                        full_path = os.path.join(root_dir, filename)
                        # Return relative path for cleaner output
                        rel_path = os.path.relpath(full_path, search_root)
                        matches.append(rel_path)
        else:
            # Non-recursive search
            for entry in os.listdir(search_root):
                full_path = os.path.join(search_root, entry)
                if os.path.isfile(full_path) and not entry.startswith('.'):
                    if fnmatch.fnmatch(entry, pattern):
                        matches.append(entry)
        
        # Sort results for consistent output
        matches.sort()
        
        if not matches:
            return f"No files matching '{pattern}' found in {search_root}"
        
        # Format output
        result_lines = [
            f"Found {len(matches)} file(s) matching '{pattern}' in {search_root}:",
            "",
        ]
        result_lines.extend(matches)
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"Error during search: {e}"
