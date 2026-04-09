"""
File search tool: search for files by name or content pattern.

Provides grep-like functionality to find files matching name patterns
or containing specific text content within the allowed root directory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_search",
        "description": (
            "Search for files by name pattern or content. "
            "Uses find and grep for efficient searching. "
            "Results are limited to prevent overwhelming output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename glob or text to search for).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["filename", "content"],
                    "description": "Type of search: 'filename' for file names, 'content' for file contents.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (defaults to allowed root).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
            },
            "required": ["pattern", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for files by name or content.
    
    Args:
        pattern: The search pattern (glob for filenames, regex for content)
        search_type: Either 'filename' or 'content'
        path: Directory to search (defaults to allowed root)
        max_results: Maximum results to return
    
    Returns:
        String with search results or error message
    """
    # Validate inputs
    if not pattern or not pattern.strip():
        return "Error: pattern is required"
    
    if search_type not in ("filename", "content"):
        return f"Error: search_type must be 'filename' or 'content', got '{search_type}'"
    
    # Determine search path
    search_path = path or _ALLOWED_ROOT or os.getcwd()
    search_path = os.path.abspath(search_path)
    
    # Validate path is within allowed root
    if _ALLOWED_ROOT is not None:
        if not search_path.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Path '{search_path}' is outside allowed root '{_ALLOWED_ROOT}'"
    
    if not os.path.isdir(search_path):
        return f"Error: '{search_path}' is not a valid directory"
    
    try:
        if search_type == "filename":
            return _search_by_filename(pattern, search_path, max_results)
        else:
            return _search_by_content(pattern, search_path, max_results)
    except Exception as e:
        return f"Error during search: {type(e).__name__}: {e}"


def _search_by_filename(pattern: str, search_path: str, max_results: int) -> str:
    """Search for files by name pattern using find command."""
    # Escape the pattern for shell safety
    escaped_pattern = pattern.replace("'", "'\"'\"'")
    
    # Use find command with -name for glob matching
    cmd = [
        "find", search_path,
        "-type", "f",
        "-name", pattern,
        "-not", "-path", "*/\.*",  # Exclude hidden files/dirs
        "-not", "-path", "*/__pycache__/*",
        "-not", "-path", "*/node_modules/*",
        "-not", "-path", "*/.git/*",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error: find command failed: {result.stderr}"
        
        files = [f for f in result.stdout.strip().split("\n") if f]
        
        if not files:
            return f"No files matching '{pattern}' found in {search_path}"
        
        # Limit results
        total = len(files)
        if total > max_results:
            files = files[:max_results]
            truncated_msg = f"\n... ({total - max_results} more results truncated)"
        else:
            truncated_msg = ""
        
        # Format results with relative paths
        rel_files = []
        for f in files:
            try:
                rel = os.path.relpath(f, search_path)
                rel_files.append(rel)
            except ValueError:
                rel_files.append(f)
        
        result_str = f"Found {total} file(s) matching '{pattern}':\n" + "\n".join(f"  - {f}" for f in rel_files)
        if truncated_msg:
            result_str += truncated_msg
        
        return result_str
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern."
    except Exception as e:
        return f"Error executing search: {e}"


def _search_by_content(pattern: str, search_path: str, max_results: int) -> str:
    """Search for files containing specific text using grep."""
    # Use grep with recursive search
    cmd = [
        "grep", "-r",
        "-l",  # Only list filenames
        "-i",  # Case insensitive
        "--include=*.py",
        "--include=*.txt",
        "--include=*.md",
        "--include=*.json",
        "--include=*.yaml",
        "--include=*.yml",
        "--exclude-dir=.*",
        "--exclude-dir=__pycache__",
        "--exclude-dir=node_modules",
        "--exclude-dir=.git",
        pattern,
        search_path,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in (0, 1):
            return f"Error: grep command failed: {result.stderr}"
        
        files = [f for f in result.stdout.strip().split("\n") if f]
        
        if not files:
            return f"No files containing '{pattern}' found in {search_path}"
        
        # Limit results
        total = len(files)
        if total > max_results:
            files = files[:max_results]
            truncated_msg = f"\n... ({total - max_results} more results truncated)"
        else:
            truncated_msg = ""
        
        # Format results with relative paths
        rel_files = []
        for f in files:
            try:
                rel = os.path.relpath(f, search_path)
                rel_files.append(rel)
            except ValueError:
                rel_files.append(f)
        
        result_str = f"Found {total} file(s) containing '{pattern}':\n" + "\n".join(f"  - {f}" for f in rel_files)
        if truncated_msg:
            result_str += truncated_msg
        
        return result_str
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or smaller directory."
    except Exception as e:
        return f"Error executing search: {e}"
