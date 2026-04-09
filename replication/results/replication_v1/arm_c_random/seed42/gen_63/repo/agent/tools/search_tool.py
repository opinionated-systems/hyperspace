"""
Search tool for exploring and finding files in the codebase.

Provides functionality to search for files by name pattern and
search for text content within files.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any


def _search_files_by_name(
    directory: str,
    pattern: str,
    recursive: bool = True,
) -> list[str]:
    """Search for files matching a name pattern.
    
    Args:
        directory: Root directory to search in
        pattern: Glob pattern to match (e.g., "*.py", "test_*.py")
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths (relative to directory)
    """
    matches = []
    root = Path(directory)
    
    if recursive:
        for path in root.rglob(pattern):
            if path.is_file():
                matches.append(str(path.relative_to(root)))
    else:
        for path in root.glob(pattern):
            if path.is_file():
                matches.append(str(path.relative_to(root)))
    
    return sorted(matches)


def _search_content_in_files(
    directory: str,
    pattern: str,
    file_pattern: str = "*",
    recursive: bool = True,
    case_sensitive: bool = False,
) -> list[dict[str, Any]]:
    """Search for text content within files.
    
    Args:
        directory: Root directory to search in
        pattern: Text pattern to search for (regex supported)
        file_pattern: Glob pattern for files to search (e.g., "*.py")
        recursive: Whether to search recursively
        case_sensitive: Whether the search is case sensitive
        
    Returns:
        List of matches with file path, line number, and line content
    """
    matches = []
    root = Path(directory)
    flags = 0 if case_sensitive else re.IGNORECASE
    
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return [{"error": f"Invalid regex pattern: {e}"}]
    
    # Find files to search
    if recursive:
        files = list(root.rglob(file_pattern))
    else:
        files = list(root.glob(file_pattern))
    
    for file_path in files:
        if not file_path.is_file():
            continue
            
        # Skip binary files
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append({
                            "file": str(file_path.relative_to(root)),
                            "line": line_num,
                            "content": line.rstrip("\n\r"),
                        })
        except (IOError, OSError, UnicodeDecodeError):
            # Skip files that can't be read
            continue
    
    return matches


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files by name pattern or search for text content within files in a directory. Provides two modes: 'files' to find files by name pattern, and 'content' to search for text within file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["files", "content"],
                        "description": "Search mode: 'files' to search by filename pattern, 'content' to search within file contents",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search in (absolute or relative path)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "For 'files': glob pattern like '*.py' or 'test_*.py'. For 'content': text or regex pattern to search for",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "When using 'content' mode, filter by file pattern (e.g., '*.py'). Default is '*' (all files)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively in subdirectories. Default is true",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "When using 'content' mode, whether search is case sensitive. Default is false",
                    },
                },
                "required": ["command", "directory", "pattern"],
            },
        },
    }


def tool_function(
    command: str,
    directory: str,
    pattern: str,
    file_pattern: str = "*",
    recursive: bool = True,
    case_sensitive: bool = False,
) -> str:
    """Execute search command.
    
    Args:
        command: 'files' or 'content'
        directory: Directory to search in
        pattern: Pattern to search for
        file_pattern: File pattern filter for content search
        recursive: Whether to search recursively
        case_sensitive: Whether content search is case sensitive
        
    Returns:
        JSON string with search results
    """
    import json
    
    # Expand user directory if needed
    directory = os.path.expanduser(directory)
    
    if not os.path.isdir(directory):
        return json.dumps({
            "error": f"Directory not found: {directory}",
            "results": [],
        })
    
    if command == "files":
        results = _search_files_by_name(directory, pattern, recursive)
        return json.dumps({
            "command": "files",
            "directory": directory,
            "pattern": pattern,
            "recursive": recursive,
            "count": len(results),
            "results": results,
        }, indent=2)
    
    elif command == "content":
        results = _search_content_in_files(
            directory, pattern, file_pattern, recursive, case_sensitive
        )
        # Limit results to avoid overwhelming output
        max_results = 100
        truncated = len(results) > max_results
        display_results = results[:max_results]
        
        return json.dumps({
            "command": "content",
            "directory": directory,
            "pattern": pattern,
            "file_pattern": file_pattern,
            "recursive": recursive,
            "case_sensitive": case_sensitive,
            "count": len(results),
            "truncated": truncated,
            "results": display_results,
        }, indent=2)
    
    else:
        return json.dumps({
            "error": f"Unknown command: {command}. Use 'files' or 'content'.",
            "results": [],
        })
