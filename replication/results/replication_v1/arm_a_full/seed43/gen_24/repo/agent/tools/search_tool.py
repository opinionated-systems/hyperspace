"""
Search tool: find files and search content within files.

Provides grep-like functionality for searching code and text files.
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
            "find_in_files (search text in multiple files). "
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
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (glob for find_files, regex for grep).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to filter by (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
            },
            "required": ["command", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path_allowed(path: str) -> tuple[bool, str]:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True, ""
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, ""


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        allowed, error = _check_path_allowed(path)
        if not allowed:
            return error

        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        if not p.exists():
            return f"Error: {path} does not exist."

        if command == "find_files":
            return _find_files(p, pattern or "*", max_results)
        elif command == "grep":
            if pattern is None:
                return "Error: pattern required for grep."
            return _grep(p, pattern, file_pattern, max_results)
        elif command == "find_in_files":
            if pattern is None:
                return "Error: pattern required for find_in_files."
            return _find_in_files(p, pattern, file_pattern, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(base_path: Path, pattern: str, max_results: int) -> str:
    """Find files matching a glob pattern."""
    results = []
    count = 0
    
    try:
        for item in base_path.rglob(pattern):
            if count >= max_results:
                results.append(f"... (truncated after {max_results} results)")
                break
            if item.is_file():
                results.append(str(item))
                count += 1
    except Exception as e:
        return f"Error during search: {e}"
    
    if not results:
        return f"No files found matching '{pattern}' in {base_path}"
    
    return f"Found {count} file(s) matching '{pattern}':\n" + "\n".join(results)


def _grep(base_path: Path, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Search for content matching a regex pattern in files."""
    results = []
    count = 0
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"
    
    try:
        if file_pattern:
            files = list(base_path.rglob(file_pattern))
        else:
            files = [f for f in base_path.rglob("*") if f.is_file() and not _is_binary(f)]
        
        for file_path in files:
            if count >= max_results:
                results.append(f"... (truncated after {max_results} results)")
                break
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        results.append(f"{file_path}:{i}: {line[:200]}")
                        count += 1
                        if count >= max_results:
                            break
            except Exception:
                # Skip files that can't be read
                continue
    except Exception as e:
        return f"Error during search: {e}"
    
    if not results:
        return f"No matches found for pattern '{pattern}' in {base_path}"
    
    return f"Found {count} match(es) for '{pattern}':\n" + "\n".join(results)


def _find_in_files(base_path: Path, text: str, file_pattern: str | None, max_results: int) -> str:
    """Search for literal text in files (case-insensitive)."""
    results = []
    count = 0
    
    search_text = text.lower()
    
    try:
        if file_pattern:
            files = list(base_path.rglob(file_pattern))
        else:
            files = [f for f in base_path.rglob("*") if f.is_file() and not _is_binary(f)]
        
        for file_path in files:
            if count >= max_results:
                results.append(f"... (truncated after {max_results} results)")
                break
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    if search_text in line.lower():
                        results.append(f"{file_path}:{i}: {line[:200]}")
                        count += 1
                        if count >= max_results:
                            break
            except Exception:
                # Skip files that can't be read
                continue
    except Exception as e:
        return f"Error during search: {e}"
    
    if not results:
        return f"No matches found for text '{text}' in {base_path}"
    
    return f"Found {count} match(es) for '{text}':\n" + "\n".join(results)


def _is_binary(file_path: Path) -> bool:
    """Check if a file is binary (heuristic)."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        return True
