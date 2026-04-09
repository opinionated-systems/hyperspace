"""
Search tool: grep and find functionality for searching file contents.

Provides file search capabilities to help agents locate code patterns,
function definitions, and specific text within the codebase.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


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

# Simple cache for search results: (search_key) -> (timestamp, results)
_search_cache: dict[str, tuple[float, str]] = {}
_CACHE_MAX_SIZE = 100
_CACHE_TTL_SECONDS = 60  # Cache results for 1 minute


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _get_cache_key(pattern: str, path: str | None, file_pattern: str | None, 
                   is_regex: bool, case_sensitive: bool, max_results: int) -> str:
    """Generate a cache key for the search parameters."""
    key_data = f"{pattern}:{path}:{file_pattern}:{is_regex}:{case_sensitive}:{max_results}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_cached_result(cache_key: str) -> str | None:
    """Get cached result if it exists and is not expired."""
    if cache_key not in _search_cache:
        return None
    
    timestamp, result = _search_cache[cache_key]
    if time.time() - timestamp > _CACHE_TTL_SECONDS:
        # Cache expired
        del _search_cache[cache_key]
        return None
    return result


def _cache_result(cache_key: str, result: str) -> None:
    """Cache a search result with current timestamp."""
    global _search_cache
    
    # Simple LRU: if cache is full, clear oldest entries
    if len(_search_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest 20% of entries
        sorted_items = sorted(_search_cache.items(), key=lambda x: x[1][0])
        for key, _ in sorted_items[:_CACHE_MAX_SIZE // 5]:
            del _search_cache[key]
    
    _search_cache[cache_key] = (time.time(), result)


def clear_search_cache() -> None:
    """Clear the search cache. Useful for testing or when files change."""
    global _search_cache
    _search_cache.clear()
    logger.info("Search cache cleared")


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
    """Execute a search command with optimized file traversal and caching."""
    try:
        # Validate pattern
        if not pattern or not pattern.strip():
            return "Error: pattern cannot be empty"
        
        # Determine search path
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        
        if not _is_within_allowed(search_path):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if not os.path.exists(search_path):
            return f"Error: path does not exist: {search_path}"
        
        # Check cache for repeated searches
        cache_key = _get_cache_key(pattern, search_path, file_pattern, is_regex, case_sensitive, max_results)
        cached_result = _get_cached_result(cache_key)
        if cached_result is not None:
            return f"[Cached] {cached_result}"
        
        results = []
        count = 0
        files_searched = 0
        
        # Compile regex once for efficiency
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            if is_regex:
                regex = re.compile(pattern, flags)
            else:
                # Escape special regex characters for plain text search
                escaped = re.escape(pattern)
                regex = re.compile(escaped, flags)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"
        
        # Build file list with optimized filtering
        if os.path.isfile(search_path):
            files_to_search = [Path(search_path)]
        else:
            # Use os.walk for better performance than rglob for large directories
            files_to_search = []
            skip_dirs = {'.git', '.svn', '.hg', '__pycache__', '.pytest_cache', 
                        'node_modules', '.tox', '.venv', 'venv', 'env'}
            
            for root, dirs, files in os.walk(search_path):
                # Filter out hidden and cache directories in-place for efficiency
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in skip_dirs]
                
                for filename in files:
                    # Skip hidden files
                    if filename.startswith('.'):
                        continue
                    
                    # Apply file pattern filter if specified
                    if file_pattern and not _match_pattern(filename, file_pattern):
                        continue
                    
                    files_to_search.append(Path(root) / filename)
        
        # Search files with early termination
        for file_path in files_to_search:
            if count >= max_results:
                break
            
            files_searched += 1
            
            try:
                # Skip binary files quickly
                if _is_binary(file_path):
                    continue
                
                # Read and search file with context manager
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            rel_path = os.path.relpath(file_path, search_path)
                            results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                            count += 1
                            if count >= max_results:
                                break
            except (IOError, OSError, PermissionError, UnicodeDecodeError):
                # Silently skip files we can't read
                continue
            except Exception as e:
                # Log unexpected errors but continue searching
                logger.debug(f"Error searching {file_path}: {e}")
                continue
        
        if not results:
            result = f"No matches found for pattern '{pattern}' (searched {files_searched} files)"
            _cache_result(cache_key, result)
            return result
        
        output = f"Found {count} match(es) for pattern '{pattern}' (searched {files_searched} files):\n" + "\n".join(results)
        if count >= max_results:
            output += f"\n... (results limited to {max_results} matches)"
        
        final_output = _truncate_output(output)
        _cache_result(cache_key, final_output)
        return final_output
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return f"Error during search: {e}"


def _match_pattern(filename: str, pattern: str) -> bool:
    """Check if filename matches a glob pattern."""
    return fnmatch.fnmatch(filename, pattern)


def _is_binary(file_path: Path) -> bool:
    """Check if file is binary by reading first chunk."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            # Check for null bytes or high ratio of non-printable characters
            if b'\x00' in chunk:
                return True
            # Also check for common binary file signatures
            binary_signatures = [
                b'\x89PNG',  # PNG
                b'\xff\xd8\xff',  # JPEG
                b'GIF87a', b'GIF89a',  # GIF
                b'PK\x03\x04',  # ZIP
                b'\x1f\x8b',  # GZIP
                b'\x42\x5a\x68',  # BZ2
                b'\xfd7zXZ',  # XZ
                b'\xca\xfe\xba\xbe',  # Java class
                b'MZ',  # Windows executable
                b'\x7fELF',  # ELF
            ]
            for sig in binary_signatures:
                if chunk.startswith(sig):
                    return True
            return False
    except:
        return True
