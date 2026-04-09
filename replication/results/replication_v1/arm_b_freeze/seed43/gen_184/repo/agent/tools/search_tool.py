"""
Search tool: find files and search content within the repository.

Provides grep-like functionality and file finding capabilities
to help navigate and explore codebases efficiently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content within the repository. "
            "Supports grep-style content search and file finding by name/pattern. "
            "Useful for exploring codebases and finding relevant code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (text to find or file pattern).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Type of search: 'content' for text search, 'filename' for file patterns.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to limit search (e.g., '*.py', '*.md').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether to perform case-sensitive search (default: False). Only applies to content search.",
                },
            },
            "required": ["query", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _get_search_path(path: str | None) -> str:
    """Get the effective search path, respecting allowed root."""
    if path is None:
        return _ALLOWED_ROOT or os.getcwd()
    
    resolved = os.path.abspath(path)
    if _ALLOWED_ROOT is not None:
        if not resolved.startswith(_ALLOWED_ROOT):
            return _ALLOWED_ROOT
    return resolved


def _truncate_output(output: str, max_len: int = 10000) -> str:
    """Truncate output to prevent context overflow."""
    if len(output) <= max_len:
        return output
    half = max_len // 2
    return output[:half] + "\n... [output truncated] ...\n" + output[-half:]


def _search_content(
    query: str,
    search_path: str,
    file_pattern: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = False,
) -> str:
    """Search for content within files using grep."""
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add case-insensitive flag if not case-sensitive
        if not case_sensitive:
            cmd.append("-i")
        
        cmd.extend(["--include", file_pattern or "*"])
        
        # Add exclude patterns for common non-source directories
        exclude_dirs = [
            "__pycache__", ".git", ".venv", "venv", 
            "node_modules", ".pytest_cache", ".mypy_cache"
        ]
        for d in exclude_dirs:
            cmd.extend(["--exclude-dir", d])
        
        cmd.extend([query, search_path])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"\n... and {len(result.stdout.strip().split(chr(10))) - max_results} more matches ...")
            return _truncate_output("\n".join(lines))
        elif result.returncode == 1:
            return f"No matches found for '{query}'"
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (30s limit)"
    except Exception as e:
        return f"Error during search: {type(e).__name__}: {e}"


def _search_filename(
    query: str,
    search_path: str,
    max_results: int = 50,
) -> str:
    """Search for files by name pattern."""
    try:
        # Use find command for filename search
        cmd = [
            "find", search_path,
            "-type", "f",
            "-name", query,
            "-not", "-path", "*/\.*",
            "-not", "-path", "*/__pycache__/*",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            lines = [line for line in lines if line]  # Remove empty lines
            if not lines:
                return f"No files found matching '{query}'"
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"\n... and {len([l for l in result.stdout.strip().split(chr(10)) if l]) - max_results} more files ...")
            return _truncate_output("\n".join(lines))
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (30s limit)"
    except Exception as e:
        return f"Error during search: {type(e).__name__}: {e}"


def tool_function(
    query: str,
    search_type: str,
    path: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = False,
) -> str:
    """Execute a search query.
    
    Args:
        query: Text to search for or filename pattern
        search_type: Either 'content' (grep) or 'filename' (find)
        path: Directory to search in (default: allowed root)
        file_pattern: Pattern to limit file search (e.g., '*.py')
        max_results: Maximum results to return
        case_sensitive: Whether to perform case-sensitive search (content only)
        
    Returns:
        Search results or error message
    """
    if not query or not query.strip():
        return "Error: Empty query provided"
    
    search_path = _get_search_path(path)
    
    if search_type == "content":
        return _search_content(query, search_path, file_pattern, max_results, case_sensitive)
    elif search_type == "filename":
        return _search_filename(query, search_path, max_results)
    else:
        return f"Error: Unknown search_type '{search_type}'. Use 'content' or 'filename'."
