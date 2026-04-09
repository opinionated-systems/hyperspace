"""
Search tool for finding files and content within the codebase.

Provides grep-like and find-like functionality to help the agent
locate files and code patterns efficiently.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def _search_files(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = False,
) -> str:
    """Search for files containing a pattern.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory path to search in (default: current directory)
        file_extension: Optional file extension filter (e.g., '.py')
        max_results: Maximum number of matches to return
        case_sensitive: Whether the search should be case sensitive
        
    Returns:
        String containing search results with file paths and line numbers
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."
    
    results = []
    flags = 0 if case_sensitive else re.IGNORECASE
    
    try:
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    count = 0
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', 'venv', '.git')]
        
        for filename in files:
            if filename.startswith('.'):
                continue
            if file_extension and not filename.endswith(file_extension):
                continue
                
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled_pattern.search(line):
                            results.append(f"{filepath}:{line_num}: {line.rstrip()}")
                            count += 1
                            if count >= max_results:
                                break
                    if count >= max_results:
                        break
            except (IOError, OSError, UnicodeDecodeError):
                continue
        
        if count >= max_results:
            break
    
    if not results:
        return f"No matches found for pattern '{pattern}' in '{path}'"
    
    header = f"Found {len(results)} matches for pattern '{pattern}':\n"
    return header + "\n".join(results[:max_results])


def _find_files(
    name_pattern: str,
    path: str = ".",
    max_results: int = 30,
) -> str:
    """Find files by name pattern.
    
    Args:
        name_pattern: Glob pattern for file names (e.g., '*.py', 'test_*.py')
        path: Directory path to search in (default: current directory)
        max_results: Maximum number of files to return
        
    Returns:
        String containing list of matching file paths
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."
    
    results = []
    
    # Convert glob pattern to regex
    regex_pattern = name_pattern.replace('.', r'\.').replace('*', '.*').replace('?', '.')
    regex_pattern = f"^{regex_pattern}$"
    
    try:
        compiled = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid pattern: {e}"
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', 'venv', '.git')]
        
        for filename in files:
            if compiled.match(filename):
                results.append(os.path.join(root, filename))
                if len(results) >= max_results:
                    break
        
        if len(results) >= max_results:
            break
    
    if not results:
        return f"No files found matching '{name_pattern}' in '{path}'"
    
    header = f"Found {len(results)} files matching '{name_pattern}':\n"
    return header + "\n".join(results[:max_results])


def tool_info() -> dict[str, Any]:
    """Return tool metadata for the registry."""
    return {
        "name": "search",
        "description": "Search for files and content within the codebase. Provides grep-like content search and find-like file discovery.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["search_files", "find_files"],
                    "description": "The search command to execute: 'search_files' for content search, 'find_files' for file name search",
                },
                "pattern": {
                    "type": "string",
                    "description": "For 'search_files': the regex pattern to search for in file contents",
                },
                "name_pattern": {
                    "type": "string",
                    "description": "For 'find_files': the glob pattern for file names (e.g., '*.py', 'test_*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: current directory)",
                    "default": ".",
                },
                "file_extension": {
                    "type": "string",
                    "description": "For 'search_files': optional file extension filter (e.g., '.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 50,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "For 'search_files': whether the search should be case sensitive",
                    "default": False,
                },
            },
            "required": ["command"],
        },
    }


def tool_function(
    command: str,
    path: str = ".",
    max_results: int = 50,
    pattern: str | None = None,
    name_pattern: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Execute the search tool.
    
    Args:
        command: The search command ('search_files' or 'find_files')
        path: Directory path to search in
        max_results: Maximum number of results
        pattern: Regex pattern for content search
        name_pattern: Glob pattern for file name search
        file_extension: Optional file extension filter
        case_sensitive: Whether search should be case sensitive
        
    Returns:
        Search results as a formatted string
    """
    if command == "search_files":
        if not pattern:
            return "Error: 'pattern' is required for 'search_files' command"
        return _search_files(
            pattern=pattern,
            path=path,
            file_extension=file_extension,
            max_results=max_results,
            case_sensitive=case_sensitive,
        )
    elif command == "find_files":
        if not name_pattern:
            return "Error: 'name_pattern' is required for 'find_files' command"
        return _find_files(
            name_pattern=name_pattern,
            path=path,
            max_results=max_results,
        )
    else:
        return f"Error: Unknown command '{command}'. Use 'search_files' or 'find_files'."
