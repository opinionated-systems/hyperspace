"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality to search for patterns in files,
and find files by name pattern. Useful for exploring codebases.
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
            "Search for files or content within the codebase. "
            "Supports grep-like pattern matching and file name searches."
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
                    "description": "Directory to search in (default: current directory).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File name pattern to match (e.g., '*.py', default: all files).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Type of search: 'content' searches file contents, 'filename' searches file names.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for searches."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    try:
        return os.path.abspath(path).startswith(_ALLOWED_ROOT)
    except Exception:
        return False


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    search_type: str = "content",
) -> str:
    """Search for files or content.

    Args:
        pattern: Search pattern (regex supported for content, glob for filename)
        path: Directory to search in
        file_pattern: Optional file name filter (e.g., '*.py')
        search_type: 'content' or 'filename'

    Returns:
        Search results as formatted string
    """
    search_path = os.path.abspath(path)
    
    if not _is_within_allowed(search_path):
        return f"Error: Search path '{path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{path}' does not exist."
    
    if search_type == "filename":
        return _search_filename(pattern, search_path)
    else:
        return _search_content(pattern, search_path, file_pattern)


def _search_filename(pattern: str, search_path: str) -> str:
    """Search for files by name pattern."""
    results = []
    
    # Convert glob pattern to regex
    regex_pattern = pattern.replace(".", r"\.")
    regex_pattern = regex_pattern.replace("*", ".*")
    regex_pattern = regex_pattern.replace("?", ".")
    
    try:
        compiled = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid pattern '{pattern}': {e}"
    
    for root, dirs, files in os.walk(search_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for filename in files:
            if compiled.search(filename):
                full_path = os.path.join(root, filename)
                if _is_within_allowed(full_path):
                    rel_path = os.path.relpath(full_path, search_path)
                    results.append(rel_path)
    
    if not results:
        return f"No files matching '{pattern}' found."
    
    return f"Found {len(results)} file(s):\n" + "\n".join(results[:50])  # Limit results


def _search_content(pattern: str, search_path: str, file_pattern: str | None) -> str:
    """Search for content within files using grep."""
    if not _is_within_allowed(search_path):
        return f"Error: Search path outside allowed root."
    
    # Build grep command
    cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", pattern, search_path]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Format and limit results
            formatted = []
            for line in lines[:30]:  # Limit to 30 matches
                # Remove the search_path prefix for cleaner output
                if line.startswith(search_path):
                    line = line[len(search_path):].lstrip("/")
                formatted.append(line)
            
            output = "\n".join(formatted)
            if len(lines) > 30:
                output += f"\n... and {len(lines) - 30} more matches"
            return output
        elif result.returncode == 1:
            return f"No matches found for '{pattern}'."
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (too many files or complex pattern)."
    except Exception as e:
        return f"Error during search: {e}"
