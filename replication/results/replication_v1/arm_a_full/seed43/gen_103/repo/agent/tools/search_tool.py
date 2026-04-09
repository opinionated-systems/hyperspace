"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content. "
            "Commands: find_files (glob pattern), grep (search content), "
            "find_in_files (search text in files). "
            "Useful for locating code patterns or specific files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (glob for find_files, regex for grep).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to limit search (e.g., '*.py' for grep).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["command", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def _truncate_results(results: list[str], max_results: int = 50) -> str:
    """Truncate results list to max_results."""
    if len(results) > max_results:
        truncated = results[:max_results]
        return "\n".join(truncated) + f"\n... ({len(results) - max_results} more results truncated)"
    return "\n".join(results)


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        search_path = path or "."
        
        # Scope check
        if not _check_path_allowed(search_path):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if command == "find_files":
            return _find_files(pattern, search_path, max_results)
        elif command == "grep":
            return _grep(pattern, search_path, file_pattern, max_results)
        elif command == "find_in_files":
            return _find_in_files(pattern, search_path, file_pattern, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(pattern: str, path: str, max_results: int) -> str:
    """Find files matching a glob pattern."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: path {path} does not exist"
        
        matches = list(p.rglob(pattern))
        # Filter out hidden directories and files
        matches = [m for m in matches if not any(part.startswith('.') for part in m.parts)]
        
        if not matches:
            return f"No files matching '{pattern}' found in {path}"
        
        results = [str(m) for m in matches]
        return f"Found {len(results)} files:\n" + _truncate_results(results, max_results)
    except Exception as e:
        return f"Error finding files: {e}"


def _grep(pattern: str, path: str, file_pattern: str | None, max_results: int) -> str:
    """Search for regex pattern in file contents."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: path {path} does not exist"
        
        results = []
        file_glob = file_pattern or "*"
        
        for file_path in p.rglob(file_glob):
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    try:
                        if re.search(pattern, line):
                            results.append(f"{file_path}:{i}:{line}")
                            if len(results) >= max_results * 2:  # Buffer for truncation
                                break
                    except re.error:
                        continue
                if len(results) >= max_results * 2:
                    break
            except (IOError, OSError):
                continue
        
        if not results:
            return f"No matches for pattern '{pattern}' in {path}"
        
        return f"Found {len(results)} matches:\n" + _truncate_results(results, max_results)
    except Exception as e:
        return f"Error in grep: {e}"


def _find_in_files(text: str, path: str, file_pattern: str | None, max_results: int) -> str:
    """Search for literal text in file contents (case-insensitive)."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: path {path} does not exist"
        
        results = []
        file_glob = file_pattern or "*"
        text_lower = text.lower()
        
        for file_path in p.rglob(file_glob):
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if text_lower in content.lower():
                    # Find line numbers
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if text_lower in line.lower():
                            results.append(f"{file_path}:{i}:{line}")
                            if len(results) >= max_results * 2:
                                break
                if len(results) >= max_results * 2:
                    break
            except (IOError, OSError):
                continue
        
        if not results:
            return f"No matches for text '{text}' in {path}"
        
        return f"Found {len(results)} matches:\n" + _truncate_results(results, max_results)
    except Exception as e:
        return f"Error in find_in_files: {e}"
