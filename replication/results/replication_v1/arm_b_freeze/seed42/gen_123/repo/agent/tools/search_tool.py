"""
Search tool: find files and search content within the codebase.

Provides grep-like and find-like functionality for navigating codebases.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    return {
        "name": "search",
        "description": "Search for files and content within the codebase. Supports: (1) find files by name pattern, (2) grep for content patterns, (3) list directory contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find", "grep", "ls"],
                    "description": "Search command: 'find' for files by name, 'grep' for content, 'ls' for directory listing",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: current directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (file name pattern for 'find', regex for 'grep')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively (default: true for find/grep, false for ls)",
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
    recursive: bool = True,
    max_results: int = 50,
) -> str:
    """Execute search command."""
    if command == "ls":
        return _list_directory(path, recursive, max_results)
    elif command == "find":
        return _find_files(path, pattern, recursive, max_results)
    elif command == "grep":
        return _grep_content(path, pattern, recursive, max_results)
    else:
        return f"Error: Unknown command '{command}'"


def _list_directory(path: str, recursive: bool, max_results: int) -> str:
    """List directory contents."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a directory"

    results = []
    count = 0

    if recursive:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                if f.startswith(".") or f.endswith(".pyc"):
                    continue
                full_path = os.path.join(root, f)
                results.append(full_path)
                count += 1
                if count >= max_results:
                    results.append(f"... (truncated at {max_results} results)")
                    break
            if count >= max_results:
                break
    else:
        try:
            entries = os.listdir(path)
            for entry in sorted(entries):
                if entry.startswith(".") or entry == "__pycache__":
                    continue
                full_path = os.path.join(path, entry)
                results.append(full_path + ("/" if os.path.isdir(full_path) else ""))
                count += 1
                if count >= max_results:
                    results.append(f"... (truncated at {max_results} results)")
                    break
        except Exception as e:
            return f"Error listing directory: {e}"

    return "\n".join(results) if results else "(empty directory)"


def _find_files(path: str, pattern: str, recursive: bool, max_results: int) -> str:
    """Find files by name pattern."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    results = []
    count = 0
    pattern_lower = pattern.lower()

    if recursive:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                if pattern_lower in f.lower():
                    results.append(os.path.join(root, f))
                    count += 1
                    if count >= max_results:
                        results.append(f"... (truncated at {max_results} results)")
                        break
            if count >= max_results:
                break
    else:
        try:
            for entry in os.listdir(path):
                if os.path.isfile(os.path.join(path, entry)):
                    if pattern_lower in entry.lower():
                        results.append(os.path.join(path, entry))
                        count += 1
                        if count >= max_results:
                            results.append(f"... (truncated at {max_results} results)")
                            break
        except Exception as e:
            return f"Error finding files: {e}"

    return "\n".join(results) if results else f"No files matching '{pattern}' found"


def _grep_content(path: str, pattern: str, recursive: bool, max_results: int) -> str:
    """Search file contents for pattern."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    results = []
    count = 0

    try:
        if recursive and os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                for f in files:
                    if f.endswith(".pyc") or f.startswith("."):
                        continue
                    file_path = os.path.join(root, f)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                            for i, line in enumerate(fp, 1):
                                if pattern in line:
                                    results.append(f"{file_path}:{i}:{line.rstrip()}")
                                    count += 1
                                    if count >= max_results:
                                        results.append(f"... (truncated at {max_results} results)")
                                        break
                            if count >= max_results:
                                break
                    except Exception:
                        pass
                if count >= max_results:
                    break
        elif os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                for i, line in enumerate(fp, 1):
                    if pattern in line:
                        results.append(f"{path}:{i}:{line.rstrip()}")
                        count += 1
                        if count >= max_results:
                            results.append(f"... (truncated at {max_results} results)")
                            break
    except Exception as e:
        return f"Error searching content: {e}"

    return "\n".join(results) if results else f"No matches for '{pattern}'"
