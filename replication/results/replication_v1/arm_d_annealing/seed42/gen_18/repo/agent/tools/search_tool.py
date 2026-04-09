"""
Search tool: search for files and content within the codebase.

Provides grep-like functionality to find files and search content.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for files and content within the codebase. Supports finding files by pattern and searching content with regex.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find", "grep"],
                    "description": "Search command: 'find' to list files, 'grep' to search content"
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in"
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern (for find) or regex pattern (for grep)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to filter grep results (e.g., '*.py')"
                }
            },
            "required": ["command", "path", "pattern"]
        }
    }


def _find_files(path: str, pattern: str) -> str:
    """Find files matching pattern in path."""
    if not os.path.exists(path):
        return f"Error: Path not found: {path}"
    
    matches = []
    try:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if not file.startswith('.'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, path)
                    if re.search(pattern, rel_path, re.IGNORECASE):
                        matches.append(rel_path)
    except Exception as e:
        return f"Error searching: {e}"
    
    if not matches:
        return f"No files matching '{pattern}' found in {path}"
    
    return "\n".join(matches[:50])  # Limit results


def _grep_content(path: str, pattern: str, file_pattern: str | None = None) -> str:
    """Search for regex pattern in file contents."""
    if not os.path.exists(path):
        return f"Error: Path not found: {path}"
    
    matches = []
    try:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.startswith('.'):
                    continue
                if file_pattern and not re.search(file_pattern.replace('*', '.*'), file):
                    continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, path)
                
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if re.search(pattern, line):
                                matches.append(f"{rel_path}:{i}: {line.rstrip()}")
                                if len(matches) >= 100:
                                    break
                except Exception:
                    continue
                
                if len(matches) >= 100:
                    break
            if len(matches) >= 100:
                break
    except Exception as e:
        return f"Error searching: {e}"
    
    if not matches:
        return f"No matches for '{pattern}' found in {path}"
    
    return "\n".join(matches)


def tool_function(command: str, path: str, pattern: str, file_pattern: str | None = None) -> str:
    """Execute search command."""
    if command == "find":
        return _find_files(path, pattern)
    elif command == "grep":
        return _grep_content(path, pattern, file_pattern)
    else:
        return f"Error: Unknown command '{command}'. Use 'find' or 'grep'."
