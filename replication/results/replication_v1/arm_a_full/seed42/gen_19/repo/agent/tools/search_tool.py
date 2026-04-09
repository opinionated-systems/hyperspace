"""
Search tool: find files and search content within the codebase.

Provides capabilities to:
- Find files by name pattern
- Search for text content within files
- List directory contents
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files and content within the codebase. Supports finding files by pattern, searching text content, and listing directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["find", "grep", "list"],
                        "description": "Search command: 'find' searches for files by name pattern, 'grep' searches file contents for text, 'list' lists directory contents",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (for find/list) or file/directory path to search within (for grep)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (file name pattern for 'find', regex pattern for 'grep'). Not used for 'list'.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively in subdirectories. Default is True.",
                        "default": True,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default is 50.",
                        "default": 50,
                    },
                },
                "required": ["command", "path"],
            },
        },
    }


def _find_files(root_path: str, pattern: str, recursive: bool, max_results: int) -> list[str]:
    """Find files matching the given name pattern."""
    results = []
    root = Path(root_path)
    
    if not root.exists():
        return []
    
    # Convert glob pattern to regex
    regex_pattern = pattern.replace(".", r"\.")
    regex_pattern = regex_pattern.replace("*", ".*")
    regex_pattern = regex_pattern.replace("?", ".")
    regex = re.compile(regex_pattern, re.IGNORECASE)
    
    if recursive:
        for path in root.rglob("*"):
            if path.is_file() and regex.search(path.name):
                results.append(str(path))
                if len(results) >= max_results:
                    break
    else:
        for path in root.iterdir():
            if path.is_file() and regex.search(path.name):
                results.append(str(path))
                if len(results) >= max_results:
                    break
    
    return results


def _grep_files(root_path: str, pattern: str, recursive: bool, max_results: int) -> list[dict]:
    """Search file contents for the given regex pattern."""
    results = []
    root = Path(root_path)
    
    if not root.exists():
        return []
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # If invalid regex, treat as literal string
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    
    files_to_search = []
    
    if root.is_file():
        files_to_search = [root]
    elif recursive:
        for path in root.rglob("*"):
            if path.is_file():
                files_to_search.append(path)
    else:
        for path in root.iterdir():
            if path.is_file():
                files_to_search.append(path)
    
    for file_path in files_to_search:
        # Skip binary files
        if _is_binary(file_path):
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append({
                            "file": str(file_path),
                            "line": line_num,
                            "content": line.rstrip(),
                        })
                        if len(results) >= max_results:
                            return results
        except (IOError, OSError):
            continue
    
    return results


def _is_binary(file_path: Path) -> bool:
    """Check if a file is binary by reading first few bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\0" in chunk
    except:
        return True


def _list_directory(dir_path: str, recursive: bool, max_results: int) -> list[dict]:
    """List directory contents."""
    results = []
    root = Path(dir_path)
    
    if not root.exists():
        return []
    
    if not root.is_dir():
        return []
    
    items = []
    if recursive:
        for path in root.rglob("*"):
            items.append(path)
            if len(items) >= max_results:
                break
    else:
        for path in root.iterdir():
            items.append(path)
            if len(items) >= max_results:
                break
    
    for path in items:
        item_info = {
            "path": str(path),
            "type": "directory" if path.is_dir() else "file",
        }
        if path.is_file():
            try:
                item_info["size"] = path.stat().st_size
            except OSError:
                pass
        results.append(item_info)
    
    return results


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
    recursive: bool = True,
    max_results: int = 50,
    **kwargs: Any,
) -> str:
    """Execute the search tool.
    
    Args:
        command: The search command ('find', 'grep', or 'list')
        path: The path to search in
        pattern: The search pattern (for find/grep)
        recursive: Whether to search recursively
        max_results: Maximum number of results
        
    Returns:
        JSON string with search results
    """
    import json
    
    if command == "find":
        if pattern is None:
            return json.dumps({"error": "Pattern is required for 'find' command"})
        results = _find_files(path, pattern, recursive, max_results)
        return json.dumps({
            "command": "find",
            "path": path,
            "pattern": pattern,
            "count": len(results),
            "results": results,
        }, indent=2)
    
    elif command == "grep":
        if pattern is None:
            return json.dumps({"error": "Pattern is required for 'grep' command"})
        results = _grep_files(path, pattern, recursive, max_results)
        return json.dumps({
            "command": "grep",
            "path": path,
            "pattern": pattern,
            "count": len(results),
            "results": results,
        }, indent=2)
    
    elif command == "list":
        results = _list_directory(path, recursive, max_results)
        return json.dumps({
            "command": "list",
            "path": path,
            "count": len(results),
            "results": results,
        }, indent=2)
    
    else:
        return json.dumps({"error": f"Unknown command: {command}"})
