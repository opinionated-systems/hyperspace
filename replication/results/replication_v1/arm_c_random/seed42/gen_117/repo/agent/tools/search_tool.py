"""
Search tool for finding files and content within the codebase.

Provides grep-like and find-like functionality to help the agent
locate files and search for patterns in the codebase.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for the search tool."""
    return {
        "name": "search",
        "description": "Search for files and content within the codebase. Supports finding files by name pattern and searching for text patterns within files (like grep).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find", "grep"],
                    "description": "Search command: 'find' to locate files by name pattern, 'grep' to search for text patterns within files",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (absolute or relative to repo root)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for. For 'find': glob pattern like '*.py'. For 'grep': regex pattern to search within file contents",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                    "default": 50,
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


def _find_files(path: str, pattern: str, max_results: int) -> str:
    """Find files matching a glob pattern."""
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Path does not exist: {path}"
        if not base_path.is_dir():
            return f"Error: Path is not a directory: {path}"

        matches = []
        for item in base_path.rglob(pattern):
            if len(matches) >= max_results:
                break
            try:
                # Get relative path for cleaner output
                rel_path = item.relative_to(base_path)
                matches.append(str(rel_path))
            except ValueError:
                matches.append(str(item))

        if not matches:
            return f"No files found matching pattern '{pattern}' in {path}"

        result = f"Found {len(matches)} file(s) matching '{pattern}':\n"
        result += "\n".join(matches)
        if len(matches) >= max_results:
            result += f"\n... (truncated to {max_results} results)"
        return result

    except Exception as e:
        return f"Error during find: {e}"


def _grep_files(path: str, pattern: str, max_results: int) -> str:
    """Search for pattern in file contents."""
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Path does not exist: {path}"
        if not base_path.is_dir():
            return f"Error: Path is not a directory: {path}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []
        files_searched = 0

        for item in base_path.rglob("*"):
            if item.is_file() and not _is_binary_file(item):
                files_searched += 1
                try:
                    with open(item, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                try:
                                    rel_path = item.relative_to(base_path)
                                except ValueError:
                                    rel_path = item
                                matches.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                                if len(matches) >= max_results:
                                    break
                            if len(matches) >= max_results:
                                break
                except (IOError, OSError, PermissionError):
                    continue

            if len(matches) >= max_results:
                break

        if not matches:
            return f"No matches found for pattern '{pattern}' in {path} (searched {files_searched} files)"

        result = f"Found {len(matches)} match(es) for '{pattern}' (searched {files_searched} files):\n"
        result += "\n".join(matches)
        if len(matches) >= max_results:
            result += f"\n... (truncated to {max_results} results)"
        return result

    except Exception as e:
        return f"Error during grep: {e}"


def _is_binary_file(path: Path) -> bool:
    """Check if a file is binary by reading first chunk."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\0" in chunk
    except:
        return True


def tool_function(
    command: str,
    path: str,
    pattern: str,
    max_results: int = 50,
) -> str:
    """Execute the search tool.

    Args:
        command: 'find' or 'grep'
        path: Directory to search in
        pattern: Pattern to search for
        max_results: Maximum results to return

    Returns:
        Search results as a string
    """
    if command == "find":
        return _find_files(path, pattern, max_results)
    elif command == "grep":
        return _grep_files(path, pattern, max_results)
    else:
        return f"Error: Unknown command '{command}'. Use 'find' or 'grep'."
