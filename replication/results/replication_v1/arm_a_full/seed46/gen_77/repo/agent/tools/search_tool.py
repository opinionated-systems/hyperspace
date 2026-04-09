"""
Search tool: grep and find functionality for searching file contents.

Provides file search capabilities to help agents locate code patterns,
function definitions, and specific text within the codebase.
Includes file content caching for improved performance on repeated searches.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from functools import lru_cache


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep-like functionality. "
            "Can search for text patterns, function definitions, or specific file types. "
            "Results are limited to avoid overwhelming output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex or plain text).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path). Default: allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', '*.js'). Default: all files.",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Whether pattern is a regex (true) or plain text (false). Default: false.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None
_MAX_OUTPUT_LEN = 10000

# Simple file content cache to improve performance
_file_cache: dict[str, tuple[str, float]] = {}
_MAX_CACHE_SIZE = 100
_CACHE_MAX_AGE = 60.0  # seconds


def _get_cached_file_content(file_path: str) -> str | None:
    """Get file content from cache if available and not expired."""
    if file_path in _file_cache:
        content, timestamp = _file_cache[file_path]
        if time.time() - timestamp < _CACHE_MAX_AGE:
            return content
        # Expired, remove from cache
        del _file_cache[file_path]
    return None


def _cache_file_content(file_path: str, content: str) -> None:
    """Cache file content with timestamp."""
    global _file_cache
    # Simple LRU: if cache is full, clear half of it
    if len(_file_cache) >= _MAX_CACHE_SIZE:
        # Remove oldest entries
        sorted_items = sorted(_file_cache.items(), key=lambda x: x[1][1])
        for key, _ in sorted_items[:_MAX_CACHE_SIZE // 2]:
            del _file_cache[key]
    
    _file_cache[file_path] = (content, time.time())


import time


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_allowed(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def _truncate_output(content: str, max_len: int = _MAX_OUTPUT_LEN) -> str:
    """Truncate output if too long."""
    if len(content) > max_len:
        lines = content.split("\n")
        # Show first 25 and last 25 lines (50 total)
        head_count = min(25, len(lines) // 2)
        tail_count = min(25, len(lines) // 2)
        truncated_count = len(lines) - head_count - tail_count
        return (
            "\n".join(lines[:head_count]) +
            f"\n... [{truncated_count} lines truncated] ...\n" +
            "\n".join(lines[-tail_count:])
        )
    return content


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    is_regex: bool = False,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        # Determine search path
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        
        if not _is_within_allowed(search_path):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if not os.path.exists(search_path):
            return f"Error: path does not exist: {search_path}"
        
        results = []
        count = 0
        
        # Build file list
        if os.path.isfile(search_path):
            files_to_search = [search_path]
        else:
            # Find files matching pattern
            if file_pattern:
                files_to_search = list(Path(search_path).rglob(file_pattern))
                # Filter out hidden directories and __pycache__
                files_to_search = [
                    f for f in files_to_search 
                    if not any(part.startswith(".") or part == "__pycache__" 
                              for part in f.parts)
                ]
            else:
                files_to_search = [
                    f for f in Path(search_path).rglob("*")
                    if f.is_file() and not any(
                        part.startswith(".") or part == "__pycache__"
                        for part in f.parts
                    )
                ]
        
        # Compile regex if needed
        if is_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return f"Error: invalid regex pattern: {e}"
        else:
            # Escape special regex characters for plain text search
            escaped = re.escape(pattern)
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(escaped, flags)
        
        # Search files
        for file_path in files_to_search:
            if count >= max_results:
                break
                
            try:
                # Skip binary files
                if _is_binary(file_path):
                    continue
                    
                # Try to get content from cache first
                file_path_str = str(file_path)
                content = _get_cached_file_content(file_path_str)
                
                if content is None:
                    # Read and cache file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    _cache_file_content(file_path_str, content)
                
                # Search through content line by line
                for line_num, line in enumerate(content.split('\n'), 1):
                    if regex.search(line):
                        rel_path = os.path.relpath(file_path, search_path)
                        results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                        count += 1
                        if count >= max_results:
                            break
            except (IOError, OSError, PermissionError):
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}'"
        
        output = f"Found {count} match(es) for pattern '{pattern}':\n" + "\n".join(results)
        if count >= max_results:
            output += f"\n... (results limited to {max_results} matches)"
        
        return _truncate_output(output)
        
    except Exception as e:
        return f"Error during search: {e}"


def _is_binary(file_path: Path) -> bool:
    """Check if file is binary by reading first chunk."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except:
        return True
