"""
Search tool: grep and find functionality for searching file contents.

Provides file search capabilities to help agents locate code patterns,
function definitions, and specific text within the codebase.

Features:
- Pattern matching with regex or plain text
- File type filtering
- Result caching for repeated searches
- Binary file detection and skipping
- Performance optimizations
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from functools import lru_cache
from typing import NamedTuple


class SearchCacheKey(NamedTuple):
    """Key for search result caching."""
    pattern: str
    path: str | None
    file_pattern: str | None
    is_regex: bool
    case_sensitive: bool
    max_results: int


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep-like functionality. "
            "Can search for text patterns, function definitions, or specific file types. "
            "Results are limited to avoid overwhelming output. "
            "Supports caching for repeated searches."
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
_CACHE_ENABLED = True
_search_cache: dict[SearchCacheKey, tuple[str, float]] = {}
_CACHE_MAX_SIZE = 100
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cache_key(
    pattern: str,
    path: str | None,
    file_pattern: str | None,
    is_regex: bool,
    case_sensitive: bool,
    max_results: int,
) -> SearchCacheKey:
    """Create a cache key from search parameters."""
    return SearchCacheKey(
        pattern=pattern,
        path=path,
        file_pattern=file_pattern,
        is_regex=is_regex,
        case_sensitive=case_sensitive,
        max_results=max_results,
    )


def _get_cached_result(key: SearchCacheKey) -> str | None:
    """Get cached result if it exists and is not expired."""
    if not _CACHE_ENABLED or key not in _search_cache:
        return None
    
    result, timestamp = _search_cache[key]
    if time.time() - timestamp > _CACHE_TTL_SECONDS:
        # Cache expired
        del _search_cache[key]
        return None
    
    return result


def _cache_result(key: SearchCacheKey, result: str) -> None:
    """Cache a search result with timestamp."""
    if not _CACHE_ENABLED:
        return
    
    # Simple LRU: if cache is full, clear oldest entries
    if len(_search_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest 20% of entries
        sorted_items = sorted(_search_cache.items(), key=lambda x: x[1][1])
        to_remove = int(_CACHE_MAX_SIZE * 0.2)
        for key_to_remove, _ in sorted_items[:to_remove]:
            del _search_cache[key_to_remove]
    
    _search_cache[key] = (result, time.time())


def clear_cache() -> None:
    """Clear the search cache."""
    _search_cache.clear()


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
        half_lines = len(lines) // 2
        return (
            "\n".join(lines[:half_lines]) +
            f"\n... [{len(lines) - 50} lines truncated] ...\n" +
            "\n".join(lines[-half_lines:])
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
    """Execute a search command with caching support."""
    start_time = time.time()
    
    try:
        # Validate pattern
        if not pattern or not pattern.strip():
            return "Error: pattern cannot be empty"
        
        pattern = pattern.strip()
        
        # Determine search path
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        
        if not _is_within_allowed(search_path):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if not os.path.exists(search_path):
            return f"Error: path does not exist: {search_path}"
        
        # Check cache first
        cache_key = _get_cache_key(pattern, path, file_pattern, is_regex, case_sensitive, max_results)
        cached_result = _get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result + "\n[Cached result]"
        
        results = []
        count = 0
        files_searched = 0
        files_skipped = 0
        max_files = 1000  # Limit number of files to search for performance
        
        # Build file list with optimized filtering
        if os.path.isfile(search_path):
            files_to_search = [Path(search_path)]
        else:
            # Use more efficient file discovery
            files_to_search = _get_files_to_search(search_path, file_pattern, max_files)
        
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
        
        # Search files with optimized reading
        for file_path in files_to_search:
            if count >= max_results:
                break
            
            files_searched += 1
                
            try:
                # Skip binary files
                if _is_binary(file_path):
                    files_skipped += 1
                    continue
                
                # Skip very large files (>1MB)
                file_size = file_path.stat().st_size
                if file_size > 1024 * 1024:
                    files_skipped += 1
                    continue
                
                # Use memory-mapped reading for large files (>100KB)
                if file_size > 100 * 1024:
                    result = _search_large_file(file_path, regex, search_path, max_results - count)
                    results.extend(result)
                    count = len(results)
                else:
                    # Regular line-by-line search for smaller files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = os.path.relpath(file_path, search_path)
                                # Truncate very long lines
                                if len(line) > 500:
                                    line = line[:250] + "... [truncated] ..." + line[-250:]
                                results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                                count += 1
                                if count >= max_results:
                                    break
            except (IOError, OSError, PermissionError):
                files_skipped += 1
                continue
            except Exception as e:
                # Log but don't fail on individual file errors
                files_skipped += 1
                continue
        
        if not results:
            output = f"No matches found for pattern '{pattern}' (searched {files_searched} files, skipped {files_skipped})"
            _cache_result(cache_key, output)
            return output
        
        elapsed = time.time() - start_time
        output = f"Found {count} match(es) for pattern '{pattern}' (searched {files_searched} files, skipped {files_skipped}, {elapsed:.2f}s):\n" + "\n".join(results)
        if count >= max_results:
            output += f"\n... (results limited to {max_results} matches)"
        
        result = _truncate_output(output)
        _cache_result(cache_key, result)
        return result
        
    except Exception as e:
        return f"Error during search: {e}"


def _get_files_to_search(search_path: str, file_pattern: str | None, max_files: int) -> list[Path]:
    """Get list of files to search with efficient filtering."""
    files_to_search = []
    
    # Define directories to skip
    skip_dirs = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', 'node_modules', '.venv', 'venv'}
    
    if file_pattern:
        # Use rglob for pattern matching
        for f in Path(search_path).rglob(file_pattern):
            # Check if any parent directory should be skipped
            if any(part in skip_dirs or part.startswith('.') for part in f.parts[:-1]):
                continue
            if f.is_file():
                files_to_search.append(f)
                if len(files_to_search) >= max_files:
                    break
    else:
        # Walk directory tree manually for better control
        for root, dirs, files in os.walk(search_path):
            # Filter out skip directories in-place for efficiency
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            
            for filename in files:
                if filename.startswith('.'):
                    continue
                file_path = Path(root) / filename
                if file_path.is_file():
                    files_to_search.append(file_path)
                    if len(files_to_search) >= max_files:
                        return files_to_search
    
    return files_to_search


def _search_large_file(file_path: Path, regex: re.Pattern, search_path: str, max_results: int) -> list[str]:
    """Search a large file using chunked reading for memory efficiency."""
    results = []
    chunk_size = 8192  # 8KB chunks
    overlap = 1024  # Overlap to handle matches across chunk boundaries
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            line_num = 0
            buffer = ""
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                buffer += chunk
                lines = buffer.split('\n')
                
                # Process all complete lines except the last (might be incomplete)
                for line in lines[:-1]:
                    line_num += 1
                    if regex.search(line):
                        rel_path = os.path.relpath(file_path, search_path)
                        if len(line) > 500:
                            line = line[:250] + "... [truncated] ..." + line[-250:]
                        results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                        if len(results) >= max_results:
                            return results
                
                # Keep the last line (might be incomplete) plus overlap
                buffer = lines[-1]
                if len(buffer) > overlap:
                    buffer = buffer[-overlap:]
            
            # Process any remaining content
            if buffer:
                line_num += 1
                if regex.search(buffer):
                    rel_path = os.path.relpath(file_path, search_path)
                    if len(buffer) > 500:
                        buffer = buffer[:250] + "... [truncated] ..." + buffer[-250:]
                    results.append(f"{rel_path}:{line_num}: {buffer.rstrip()}")
    
    except Exception:
        pass
    
    return results


def _is_binary(file_path: Path) -> bool:
    """Check if file is binary by reading first chunk.
    
    Uses null byte detection as a reliable indicator of binary files.
    Also handles permission errors and other edge cases gracefully.
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            # Null bytes indicate binary content
            if b'\x00' in chunk:
                return True
            # High ratio of non-printable characters also suggests binary
            if chunk:
                non_printable = sum(1 for b in chunk if b < 32 and b not in (9, 10, 13))
                return non_printable / len(chunk) > 0.3
            return False
    except (IOError, OSError, PermissionError):
        return True
    except Exception:
        return True
