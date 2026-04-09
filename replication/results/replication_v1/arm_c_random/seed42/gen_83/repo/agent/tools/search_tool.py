"""
Search tool: find files and search content within the codebase.

Provides grep-like file content search and file finding capabilities.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files or content within the codebase. Supports finding files by name pattern and searching file contents with regex or literal text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["find", "grep"],
                        "description": "Search command: 'find' to locate files by name pattern, 'grep' to search file contents",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (relative or absolute)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for. For 'find': filename pattern (e.g., '*.py'). For 'grep': text or regex pattern to search within files",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively in subdirectories (default: true)",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "For grep: whether the search is case sensitive (default: false)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50)",
                    },
                },
                "required": ["command", "path", "pattern"],
            },
        },
    }


def _validate_path(path: str) -> str:
    """Validate and normalize path, ensuring it's within allowed boundaries."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    # Prevent path traversal attacks
    if ".." in path.replace("\\", "/").split("/"):
        raise ValueError(f"Invalid path: {path}")
    return abs_path


def _find_files(path: str, pattern: str, recursive: bool = True, max_results: int = 50) -> list[str]:
    """Find files matching the given pattern."""
    results = []
    abs_path = _validate_path(path)
    
    if not os.path.isdir(abs_path):
        return [f"Error: Directory not found: {path}"]
    
    try:
        if recursive:
            for root, dirs, files in os.walk(abs_path):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for filename in files:
                    if filename.startswith('.'):
                        continue
                    # Simple glob-like matching
                    if _match_pattern(filename, pattern):
                        results.append(os.path.join(root, filename))
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
        else:
            for filename in os.listdir(abs_path):
                if filename.startswith('.'):
                    continue
                filepath = os.path.join(abs_path, filename)
                if os.path.isfile(filepath) and _match_pattern(filename, pattern):
                    results.append(filepath)
                    if len(results) >= max_results:
                        break
    except Exception as e:
        return [f"Error during find: {e}"]
    
    return results if results else [f"No files matching '{pattern}' found in {path}"]


def _match_pattern(filename: str, pattern: str) -> bool:
    """Match filename against a glob-like pattern."""
    # Convert glob pattern to regex
    regex_pattern = pattern.replace(".", r"\.")
    regex_pattern = regex_pattern.replace("*", ".*")
    regex_pattern = regex_pattern.replace("?", ".")
    regex_pattern = f"^{regex_pattern}$"
    return bool(re.match(regex_pattern, filename, re.IGNORECASE))


def _grep_files(path: str, pattern: str, recursive: bool = True, 
                case_sensitive: bool = False, max_results: int = 50) -> list[str]:
    """Search file contents for the given pattern."""
    results = []
    abs_path = _validate_path(path)
    
    if not os.path.isdir(abs_path):
        return [f"Error: Directory not found: {path}"]
    
    flags = 0 if case_sensitive else re.IGNORECASE
    
    try:
        if recursive:
            for root, dirs, files in os.walk(abs_path):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for filename in files:
                    if filename.startswith('.'):
                        continue
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if re.search(pattern, line, flags):
                                    results.append(f"{filepath}:{line_num}: {line.rstrip()}")
                                    if len(results) >= max_results:
                                        break
                    except Exception:
                        continue
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break
        else:
            for filename in os.listdir(abs_path):
                if filename.startswith('.'):
                    continue
                filepath = os.path.join(abs_path, filename)
                if os.path.isfile(filepath):
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if re.search(pattern, line, flags):
                                    results.append(f"{filepath}:{line_num}: {line.rstrip()}")
                                    if len(results) >= max_results:
                                        break
                    except Exception:
                        continue
                if len(results) >= max_results:
                    break
    except Exception as e:
        return [f"Error during grep: {e}"]
    
    return results if results else [f"No matches for '{pattern}' found in {path}"]


def tool_function(
    command: str,
    path: str,
    pattern: str,
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute search command.
    
    Args:
        command: 'find' for file search, 'grep' for content search
        path: Directory to search in
        pattern: Pattern to search for
        recursive: Whether to search subdirectories
        case_sensitive: For grep: case sensitivity
        max_results: Maximum results to return
    
    Returns:
        Search results as formatted string
    """
    max_results = min(max_results, 100)  # Cap at 100 results
    
    if command == "find":
        results = _find_files(path, pattern, recursive, max_results)
    elif command == "grep":
        results = _grep_files(path, pattern, recursive, case_sensitive, max_results)
    else:
        return f"Error: Unknown command '{command}'. Use 'find' or 'grep'."
    
    if not results:
        return "No results found."
    
    # Format results
    output = f"Found {len(results)} result(s):\n\n"
    output += "\n".join(results[:max_results])
    
    if len(results) > max_results:
        output += f"\n\n... and {len(results) - max_results} more results (truncated)"
    
    return output
