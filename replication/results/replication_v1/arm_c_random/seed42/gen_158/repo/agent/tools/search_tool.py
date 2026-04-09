"""
Search tool for finding files and content within the codebase.

Provides grep-like and find-like functionality to help the agent
explore and understand the codebase structure.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for files and content within the codebase. Supports: (1) search_files: find files by name pattern, (2) search_content: grep for text content in files, (3) list_directory: list files in a directory, (4) count_lines: count lines in files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["search_files", "search_content", "list_directory", "count_lines"],
                    "description": "The search command to execute",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: current directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename pattern for search_files, regex for search_content)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                },
            },
            "required": ["command"],
        },
    }


def tool_function(
    command: str,
    path: str = ".",
    pattern: str = "",
    file_extension: str = "",
    max_results: int = 50,
) -> str:
    """Execute search command."""
    if command == "search_files":
        return _search_files(path, pattern, max_results)
    elif command == "search_content":
        return _search_content(path, pattern, file_extension, max_results)
    elif command == "list_directory":
        return _list_directory(path, max_results)
    elif command == "count_lines":
        return _count_lines(path, pattern, file_extension, max_results)
    else:
        return f"Error: Unknown command '{command}'"


def _search_files(directory: str, pattern: str, max_results: int) -> str:
    """Find files by name pattern."""
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found"
    
    results = []
    count = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            if pattern.lower() in filename.lower():
                full_path = os.path.join(root, filename)
                results.append(full_path)
                count += 1
                if count >= max_results:
                    results.append(f"... (truncated to {max_results} results)")
                    return "\n".join(results)
    
    if not results:
        return f"No files matching '{pattern}' found in '{directory}'"
    
    return f"Found {len(results)} file(s):\n" + "\n".join(results)


def _search_content(directory: str, pattern: str, file_extension: str, max_results: int) -> str:
    """Search for content within files using regex."""
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found"
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    results = []
    count = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            # Skip binary files and hidden files
            if filename.startswith('.'):
                continue
            
            # Filter by extension if specified
            if file_extension and not filename.endswith(file_extension):
                continue
            
            full_path = os.path.join(root, filename)
            
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{full_path}:{line_num}: {line.strip()}")
                            count += 1
                            if count >= max_results:
                                results.append(f"... (truncated to {max_results} results)")
                                return "\n".join(results)
            except (IOError, OSError):
                continue
    
    if not results:
        ext_msg = f" with extension '{file_extension}'" if file_extension else ""
        return f"No matches for pattern '{pattern}'{ext_msg} in '{directory}'"
    
    return f"Found {len(results)} match(es):\n" + "\n".join(results)


def _list_directory(directory: str, max_results: int) -> str:
    """List files and directories."""
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found"
    
    try:
        entries = os.listdir(directory)
    except OSError as e:
        return f"Error: Cannot list directory '{directory}': {e}"
    
    # Separate directories and files
    dirs = []
    files = []
    
    for entry in sorted(entries):
        if entry.startswith('.'):
            continue
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            dirs.append(f"[DIR]  {entry}")
        else:
            files.append(f"[FILE] {entry}")
    
    results = dirs + files
    
    if len(results) > max_results:
        results = results[:max_results]
        results.append(f"... (truncated to {max_results} results)")
    
    if not results:
        return f"Directory '{directory}' is empty"
    
    return f"Contents of '{directory}':\n" + "\n".join(results)


def _count_lines(directory: str, pattern: str, file_extension: str, max_results: int) -> str:
    """Count lines in files matching criteria."""
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found"
    
    results = []
    total_lines = 0
    count = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            # Skip hidden files
            if filename.startswith('.'):
                continue
            
            # Filter by pattern if specified
            if pattern and pattern.lower() not in filename.lower():
                continue
            
            # Filter by extension if specified
            if file_extension and not filename.endswith(file_extension):
                continue
            
            full_path = os.path.join(root, filename)
            
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                    total_lines += lines
                    results.append((full_path, lines))
                    count += 1
                    if count >= max_results:
                        break
            except (IOError, OSError):
                continue
        
        if count >= max_results:
            break
    
    if not results:
        ext_msg = f" with extension '{file_extension}'" if file_extension else ""
        pattern_msg = f" matching '{pattern}'" if pattern else ""
        return f"No files{pattern_msg}{ext_msg} found in '{directory}'"
    
    # Sort by line count descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    lines_info = "\n".join([f"{path}: {lines} lines" for path, lines in results])
    
    if count >= max_results:
        lines_info += f"\n... (truncated to {max_results} results)"
    
    return f"Total lines: {total_lines} in {count} file(s)\n\n{lines_info}"
