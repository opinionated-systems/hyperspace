"""
Search tool: find files by name or content within the repository.

Provides grep-like functionality to search for patterns in file contents
and find files by name patterns.
Enhanced with better error handling, input validation, and result caching.
"""

from __future__ import annotations

import os
import re
import subprocess
import logging
import time
from pathlib import Path
from typing import Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# Maximum results to prevent memory issues
MAX_RESULTS_LIMIT = 1000

# Cache settings
CACHE_TTL_SECONDS = 60  # Cache results for 60 seconds
MAX_CACHE_SIZE = 128


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files by name pattern or content within the repository. "
            "Useful for finding where specific functions, classes, or patterns are defined. "
            "Results are limited to 1000 matches maximum."
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
                    "description": "Maximum number of results to return (default: 50, max: 1000).",
                },
            },
            "required": ["pattern", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None

# Simple TTL cache for search results
_search_cache: dict[tuple, tuple[str, float]] = {}


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for searches."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)
    # Clear cache when root changes
    clear_search_cache()


def clear_search_cache() -> None:
    """Clear the search result cache."""
    global _search_cache
    _search_cache.clear()
    logger.info("Search cache cleared")


def _is_within_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def _get_cache_key(pattern: str, search_type: str, path: str | None, file_extension: str | None, max_results: int) -> tuple:
    """Generate a cache key for search parameters."""
    return (pattern, search_type, path, file_extension, max_results, _ALLOWED_ROOT)


def _get_cached_result(cache_key: tuple) -> str | None:
    """Get cached result if it exists and is not expired."""
    if cache_key in _search_cache:
        result, timestamp = _search_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            logger.debug(f"Cache hit for search: {cache_key[:3]}")
            return result
        else:
            # Expired, remove from cache
            del _search_cache[cache_key]
    return None


def _set_cached_result(cache_key: tuple, result: str) -> None:
    """Cache a search result with timestamp."""
    # Simple cache eviction if too many entries
    if len(_search_cache) >= MAX_CACHE_SIZE:
        # Remove oldest entries
        sorted_items = sorted(_search_cache.items(), key=lambda x: x[1][1])
        for key, _ in sorted_items[:MAX_CACHE_SIZE // 4]:
            del _search_cache[key]
    
    _search_cache[cache_key] = (result, time.time())


def _validate_search_params(
    pattern: str,
    search_type: str,
    max_results: int,
) -> tuple[bool, str]:
    """Validate search parameters.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(pattern, str) or not pattern.strip():
        return False, "Pattern must be a non-empty string"
    
    if search_type not in ("content", "filename"):
        return False, f"search_type must be 'content' or 'filename', got '{search_type}'"
    
    if not isinstance(max_results, int) or max_results < 1:
        return False, f"max_results must be a positive integer, got {max_results}"
    
    if max_results > MAX_RESULTS_LIMIT:
        return False, f"max_results cannot exceed {MAX_RESULTS_LIMIT}, got {max_results}"
    
    # Validate regex pattern for content search
    if search_type == "content":
        try:
            re.compile(pattern)
        except re.error as e:
            return False, f"Invalid regex pattern: {e}"
    
    return True, ""


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
        max_results: Maximum results to return
    
    Returns:
        Formatted search results
    """
    # Validate parameters
    is_valid, error_msg = _validate_search_params(pattern, search_type, max_results)
    if not is_valid:
        return f"Error: {error_msg}"
    
    search_path = path or _ALLOWED_ROOT or "."
    
    if not isinstance(search_path, str):
        return f"Error: Path must be a string, got {type(search_path).__name__}"
    
    if not _is_within_allowed(search_path):
        logger.warning(f"Search path '{search_path}' outside allowed root '{_ALLOWED_ROOT}'")
        return f"Error: Search path '{search_path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist."
    
    if not os.path.isdir(search_path):
        return f"Error: Path '{search_path}' is not a directory."
    
    # Check cache first
    cache_key = _get_cache_key(pattern, search_type, search_path, file_extension, max_results)
    cached_result = _get_cached_result(cache_key)
    if cached_result is not None:
        return cached_result
    
    results: list[str] = []
    
    if search_type == "filename":
        # Use find command for filename search
        cmd = ["find", search_path, "-type", "f", "-name", pattern]
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if output.returncode != 0 and output.stderr:
                logger.warning(f"find command stderr: {output.stderr}")
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            
            if file_extension:
                files = [f for f in files if f.endswith(file_extension)]
            
            results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds."
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error in filename search: {e}")
            return f"Error: Subprocess failed - {e}"
        except Exception as e:
            logger.error(f"Unexpected error in filename search: {e}")
            return f"Error during search: {type(e).__name__}: {e}"
    
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
            # grep returns 1 when no matches found, which is not an error
            if output.returncode not in (0, 1):
                logger.warning(f"grep command returned {output.returncode}: {output.stderr}")
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds."
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error in content search: {e}")
            return f"Error: Subprocess failed - {e}"
        except Exception as e:
            logger.error(f"Unexpected error in content search: {e}")
            return f"Error during search: {type(e).__name__}: {e}"
    
    else:
        return f"Error: Invalid search_type '{search_type}'. Use 'content' or 'filename'."
    
    if not results:
        result_str = f"No results found for pattern '{pattern}' (type: {search_type})."
        _set_cached_result(cache_key, result_str)
        return result_str
    
    # Format results
    output_lines = [
        f"Found {len(results)} result(s) for pattern '{pattern}' (type: {search_type}):",
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
    
    result_str = "\n".join(output_lines)
    _set_cached_result(cache_key, result_str)
    return result_str
