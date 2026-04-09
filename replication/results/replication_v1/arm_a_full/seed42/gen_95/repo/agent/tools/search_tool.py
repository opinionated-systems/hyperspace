"""
Search tool: find files by name or content within the repository.

Provides grep-like functionality to search for patterns in file contents
and find files by name patterns.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from functools import lru_cache


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files by name pattern or content within the repository. "
            "Useful for finding where specific functions, classes, or patterns are defined."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex for content, glob for filename).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: repo root).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Whether to search in file contents or filenames.",
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
            "required": ["pattern", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None

# Simple cache for search results: (pattern, search_type, path, file_extension) -> (results, timestamp)
_search_cache: dict = {}
_CACHE_TTL = 60  # Cache results for 60 seconds


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for searches."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _get_cache_key(pattern: str, search_type: str, path: str, file_extension: str | None) -> tuple:
    """Generate a cache key for the search parameters."""
    return (pattern, search_type, path, file_extension)


def _get_cached_result(cache_key: tuple) -> list[str] | None:
    """Get cached result if it exists and is not expired."""
    if cache_key in _search_cache:
        results, timestamp = _search_cache[cache_key]
        if time.time() - timestamp < _CACHE_TTL:
            return results
        # Expired, remove from cache
        del _search_cache[cache_key]
    return None


def _cache_result(cache_key: tuple, results: list[str]) -> None:
    """Cache search results with current timestamp."""
    _search_cache[cache_key] = (results, time.time())
    # Limit cache size to prevent memory issues
    if len(_search_cache) > 100:
        # Remove oldest entries
        oldest_key = min(_search_cache.keys(), key=lambda k: _search_cache[k][1])
        del _search_cache[oldest_key]


def _is_within_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for files by name or content.
    
    Args:
        pattern: Search pattern (regex for content, glob for filename)
        search_type: "content" or "filename"
        path: Directory to search in (default: allowed root)
        file_extension: Optional extension filter (e.g., ".py")
        max_results: Maximum results to return (1-100)
    
    Returns:
        Formatted search results
    """
    # Validate inputs
    if not pattern or not isinstance(pattern, str):
        return "Error: pattern must be a non-empty string"
    
    if search_type not in ("content", "filename"):
        return f"Error: Invalid search_type '{search_type}'. Use 'content' or 'filename'."
    
    # Clamp max_results to reasonable range
    if not isinstance(max_results, int):
        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            max_results = 50
    max_results = max(1, min(100, max_results))
    
    search_path = path or _ALLOWED_ROOT or "."
    
    if not _is_within_allowed(search_path):
        return f"Error: Search path '{search_path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist."
    
    if not os.path.isdir(search_path):
        return f"Error: Path '{search_path}' is not a directory."
    
    # Check cache first
    cache_key = _get_cache_key(pattern, search_type, search_path, file_extension)
    cached_results = _get_cached_result(cache_key)
    if cached_results is not None:
        results = cached_results[:max_results]
        source = " (from cache)"
    else:
        results: list[str] = []
        source = ""
        
        if search_type == "filename":
            # Use find command for filename search
            cmd = ["find", search_path, "-type", "f", "-name", pattern]
            try:
                output = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                if output.returncode != 0 and output.stderr:
                    # find returns non-zero for some patterns but still gives results
                    pass
                files = output.stdout.strip().split("\n") if output.stdout.strip() else []
                files = [f for f in files if f and _is_within_allowed(f)]
                
                if file_extension:
                    ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
                    files = [f for f in files if f.endswith(ext)]
                
                results = files[:max_results]
                
            except subprocess.TimeoutExpired:
                return "Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
            except FileNotFoundError:
                return "Error: 'find' command not available."
            except Exception as e:
                return f"Error during filename search: {type(e).__name__}: {e}"
        
        elif search_type == "content":
            # Use grep for content search
            # Build grep command with file extension filter if provided
            include_pattern = f"*{file_extension}" if file_extension else "*"
            
            cmd = [
                "grep", "-r", "-n", "-l", "--include", include_pattern,
                pattern, search_path
            ]
            
            try:
                output = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                # grep returns 1 when no matches found, which is OK
                files = output.stdout.strip().split("\n") if output.stdout.strip() else []
                files = [f for f in files if f and _is_within_allowed(f)]
                results = files[:max_results]
                
            except subprocess.TimeoutExpired:
                return "Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
            except FileNotFoundError:
                return "Error: 'grep' command not available."
            except Exception as e:
                return f"Error during content search: {type(e).__name__}: {e}"
        
        # Cache the results
        _cache_result(cache_key, results)
    
    if not results:
        search_desc = f"pattern '{pattern}'"
        if file_extension:
            search_desc += f" with extension '{file_extension}'"
        return f"No results found for {search_desc} (type: {search_type})."
    
    # Format results
    output_lines = [
        f"Found {len(results)} result(s) for pattern '{pattern}' (type: {search_type}){source}:",
        "",
    ]
    
    for i, result in enumerate(results, 1):
        # Show relative path if within allowed root
        if _ALLOWED_ROOT and result.startswith(_ALLOWED_ROOT):
            rel_path = os.path.relpath(result, _ALLOWED_ROOT)
            output_lines.append(f"{i}. {rel_path}")
        else:
            output_lines.append(f"{i}. {result}")
    
    if len(results) == max_results:
        output_lines.append("")
        output_lines.append(f"(Results limited to {max_results}. Refine your search for more specific results.)")
    
    return "\n".join(output_lines)
