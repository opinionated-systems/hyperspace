"""
Search tool: find files and search for patterns in the codebase.

Provides grep-like functionality to search for text patterns across files,
and find files by name pattern. Useful for locating code to modify.

Recent improvements:
- Added max_results parameter to control output size
- Improved result filtering to exclude cache and hidden files
- Better error messages with context
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
            "Search for patterns in files or find files by name. "
            "Uses grep for content search and find for file search. "
            "Returns matching file paths with line numbers and context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex for content, glob for files).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: repo root).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to limit search (e.g., '*.py').",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "files"],
                    "description": "Type of search: 'content' searches file contents, 'files' searches filenames.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive (default: True).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["pattern", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for searches."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_root(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    try:
        Path(path).resolve().relative_to(Path(_ALLOWED_ROOT).resolve())
        return True
    except ValueError:
        return False


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for patterns in files or find files by name.

    Args:
        pattern: Search pattern (regex for content, glob for files)
        search_type: 'content' to search file contents, 'files' to search filenames
        path: Directory to search in (default: allowed root or current dir)
        file_pattern: File glob pattern to limit search (e.g., '*.py')
        case_sensitive: Whether search is case sensitive (default: True)
        max_results: Maximum number of results to return (default: 50)

    Returns:
        Search results with file paths and line numbers (for content search)
    """
    search_path = path or _ALLOWED_ROOT or "."
    
    if not _is_within_root(search_path):
        return f"Error: Search path '{search_path}' is outside allowed root."

    # Validate max_results
    if max_results < 1:
        max_results = 50
    if max_results > 500:
        max_results = 500  # Hard cap to prevent excessive output

    try:
        if search_type == "files":
            # Find files by name pattern
            cmd = ["find", search_path, "-type", "f", "-name", pattern]
            if not case_sensitive:
                # For case-insensitive file search, use -iname
                cmd = ["find", search_path, "-type", "f", "-iname", pattern]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            
            files = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if not files or files == [""]:
                return f"No files matching '{pattern}' found in {search_path}"
            
            # Filter out __pycache__ and hidden files
            filtered = [
                f for f in files 
                if f and "__pycache__" not in f and "/." not in f.replace(search_path, "")
            ]
            
            if not filtered:
                return f"No files matching '{pattern}' found (excluding cache/hidden files)"
            
            total = len(filtered)
            if total > max_results:
                return f"Found {total} files (showing first {max_results}):\n" + "\n".join(filtered[:max_results]) + f"\n... and {total - max_results} more files"
            
            return f"Found {total} files:\n" + "\n".join(filtered)

        elif search_type == "content":
            # Search file contents with grep
            cmd = [
                "grep", "-r", "-n", "-I",
                "--include", file_pattern or "*",
            ]
            if not case_sensitive:
                cmd.append("-i")  # Case insensitive
            cmd.extend(["-E", pattern, search_path])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # grep returns 1 when no matches found
            if result.returncode not in (0, 1):
                return f"Error: {result.stderr}"
            
            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if not lines or lines == [""]:
                return f"No matches for pattern '{pattern}' in {search_path}"
            
            # Filter out __pycache__ and binary files
            filtered = [
                line for line in lines 
                if "__pycache__" not in line and "/." not in line
            ]
            
            if not filtered:
                return f"No matches for pattern '{pattern}' (excluding cache files)"
            
            total = len(filtered)
            if total > max_results:
                return f"Found {total} matches (showing first {max_results}):\n" + "\n".join(filtered[:max_results]) + f"\n... and {total - max_results} more matches"
            
            return f"Found {total} matches:\n" + "\n".join(filtered)

        else:
            return f"Error: Unknown search_type '{search_type}'. Use 'content' or 'files'."

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
