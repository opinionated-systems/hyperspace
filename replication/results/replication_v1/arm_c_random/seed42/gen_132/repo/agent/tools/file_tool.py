"""
File tool: file metadata operations with caching.

Provides file existence checks, size queries, and directory listing.
Complements bash_tool and editor_tool.

Features:
- LRU cache for file operations to reduce redundant disk access
- Enhanced error messages with actionable suggestions
- Automatic cache invalidation on file system changes
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

# Cache configuration
_CACHE_MAXSIZE = 128
_CACHE_TTL_SECONDS = 5  # Time-to-live for cached entries

# Cache storage with timestamps: {key: (value, timestamp)}
_cache: dict[str, tuple[Any, float]] = {}
_cache_hits = 0
_cache_misses = 0


def _get_cache_key(command: str, path: str) -> str:
    """Generate a cache key for the operation."""
    return f"{command}:{path}"


def _get_cached(command: str, path: str) -> Any | None:
    """Get cached value if valid, otherwise return None."""
    global _cache_hits, _cache_misses
    key = _get_cache_key(command, path)
    if key in _cache:
        value, timestamp = _cache[key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            _cache_hits += 1
            return value
        else:
            # Expired, remove from cache
            del _cache[key]
    _cache_misses += 1
    return None


def _set_cached(command: str, path: str, value: Any) -> None:
    """Cache the result with current timestamp."""
    key = _get_cache_key(command, path)
    # Simple LRU: if cache is full, clear oldest entries
    if len(_cache) >= _CACHE_MAXSIZE:
        # Remove oldest 25% of entries
        sorted_items = sorted(_cache.items(), key=lambda x: x[1][1])
        for key_to_remove, _ in sorted_items[:_CACHE_MAXSIZE // 4]:
            del _cache[key_to_remove]
    _cache[key] = (value, time.time())


def clear_cache() -> None:
    """Clear the file operation cache."""
    _cache.clear()


def get_cache_stats() -> dict:
    """Return cache statistics."""
    total = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total if total > 0 else 0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": f"{hit_rate:.1%}",
        "size": len(_cache),
        "max_size": _CACHE_MAXSIZE,
    }


def tool_info() -> dict:
    """Return tool specification for file operations."""
    return {
        "name": "file",
        "description": "File metadata operations: check existence, get size, list directories. Features LRU caching for improved performance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "list", "is_dir", "cache_stats", "clear_cache"],
                    "description": "Operation to perform. Use 'cache_stats' to view cache performance or 'clear_cache' to reset.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path (not required for cache_stats or clear_cache)",
                },
            },
            "required": ["command"],
        },
    }


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_suggestion(path: str, p: Path) -> str:
    """Generate helpful suggestion based on path analysis."""
    suggestions = []
    
    # Check if parent exists
    if p.parent and not p.parent.exists():
        suggestions.append(f"Parent directory '{p.parent}' does not exist")
    
    # Check for common typos
    if p.parent and p.parent.exists():
        try:
            similar = [f for f in os.listdir(p.parent) 
                      if f.startswith(p.name[:3]) or p.name[:3] in f]
            if similar:
                suggestions.append(f"Did you mean: {', '.join(similar[:3])}?")
        except (PermissionError, OSError):
            pass
    
    return " | ".join(suggestions) if suggestions else ""


def tool_function(command: str, path: str = "") -> str:
    """Execute file operation with caching and enhanced error handling.

    Args:
        command: One of 'exists', 'size', 'list', 'is_dir', 'cache_stats', 'clear_cache'
        path: File or directory path (optional for cache operations)

    Returns:
        Operation result as string with helpful context
    """
    # Handle cache management commands
    if command == "cache_stats":
        stats = get_cache_stats()
        return f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']} hit rate, {stats['size']}/{stats['max_size']} entries"
    
    if command == "clear_cache":
        clear_cache()
        return "Cache cleared successfully"
    
    # Validate path for file operations
    if not path:
        return f"Error: 'path' is required for command '{command}'"
    
    p = Path(path)
    
    # Try cache first for idempotent operations
    if command in ("exists", "is_dir"):
        cached = _get_cached(command, path)
        if cached is not None:
            return "true" if cached else "false"

    if command == "exists":
        result = p.exists()
        _set_cached(command, path, result)
        return "true" if result else "false"

    elif command == "size":
        if not p.exists():
            suggestion = _get_suggestion(path, p)
            msg = f"Error: path '{path}' does not exist"
            if suggestion:
                msg += f" ({suggestion})"
            return msg
        if not p.is_file():
            if p.is_dir():
                return f"Error: '{path}' is a directory. Use 'list' command to view contents."
            return f"Error: '{path}' is not a regular file"
        try:
            size = p.stat().st_size
            return f"{_format_size(size)} ({size} bytes)"
        except (PermissionError, OSError) as e:
            return f"Error: cannot access '{path}': {e}"

    elif command == "list":
        if not p.exists():
            suggestion = _get_suggestion(path, p)
            msg = f"Error: directory '{path}' does not exist"
            if suggestion:
                msg += f" ({suggestion})"
            return msg
        if not p.is_dir():
            return f"Error: '{path}' is not a directory. Use 'size' command for file info."
        try:
            entries = os.listdir(path)
            if not entries:
                return "(empty directory)"
            # Sort entries: directories first, then files
            dirs = []
            files = []
            for entry in entries:
                full_path = p / entry
                try:
                    if full_path.is_dir():
                        dirs.append(f"{entry}/")
                    else:
                        files.append(entry)
                except (PermissionError, OSError):
                    files.append(f"{entry} (inaccessible)")
            sorted_entries = sorted(dirs, key=str.lower) + sorted(files, key=str.lower)
            return f"{len(entries)} items:\n" + "\n".join(sorted_entries)
        except PermissionError:
            return f"Error: permission denied accessing '{path}'"
        except OSError as e:
            return f"Error: cannot list directory '{path}': {e}"

    elif command == "is_dir":
        result = p.is_dir()
        _set_cached(command, path, result)
        return "true" if result else "false"

    else:
        return f"Error: unknown command '{command}'. Valid commands: exists, size, list, is_dir, cache_stats, clear_cache"
