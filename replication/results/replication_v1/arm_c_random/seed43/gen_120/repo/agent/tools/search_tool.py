"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns
within files in the repository.
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
            "Search for patterns in files. "
            "Supports regex patterns and can search recursively. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively. Default: true.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.
    
    Returns matching lines with file paths and line numbers.
    """
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No search path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        p = Path(search_path)
        if not p.exists():
            return f"Error: {search_path} does not exist."
        
        # Build file list
        if p.is_file():
            files = [p]
        else:
            if recursive:
                if file_extension:
                    files = list(p.rglob(f"*{file_extension}"))
                else:
                    files = list(p.rglob("*"))
            else:
                if file_extension:
                    files = [f for f in p.iterdir() if f.is_file() and f.suffix == file_extension]
                else:
                    files = [f for f in p.iterdir() if f.is_file()]
        
        # Filter to text files only
        files = [f for f in files if f.is_file() and not _is_binary(f)]
        
        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        
        # Search files
        matches = []
        max_matches = 100  # Limit output
        match_count = 0
        
        for f in files:
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches.append(f"{f}:{i}:{line[:200]}")
                        match_count += 1
                        if match_count >= max_matches:
                            break
                if match_count >= max_matches:
                    break
            except Exception:
                continue
        
        if not matches:
            return f"No matches found for pattern '{pattern}'"
        
        result = f"Found {len(matches)} match(es) for pattern '{pattern}':\n"
        result += "\n".join(matches)
        if match_count >= max_matches:
            result += "\n... (results truncated, showing first 100 matches)"
        
        return result
        
    except Exception as e:
        return f"Error: {e}"


def _is_binary(file_path: Path) -> bool:
    """Check if a file is binary."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        return True
