"""
Search tool: search for patterns in files and directories.

Provides grep-like functionality for finding text patterns across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterator

from agent.config import DEFAULT_AGENT_CONFIG


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files and directories. "
            "Supports regex patterns, file type filtering, and line number display. "
            "Useful for finding code references, TODOs, or specific patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to current directory.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to include (e.g., '*.py'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
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


def _is_allowed(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def _find_files(
    path: Path,
    file_pattern: str | None = None,
) -> Iterator[Path]:
    """Find files matching the pattern."""
    if path.is_file():
        yield path
        return
    
    if not path.is_dir():
        return
    
    # Use glob pattern if specified
    if file_pattern:
        yield from path.rglob(file_pattern)
    else:
        # Default: search all files but skip hidden and cache directories
        for p in path.rglob("*"):
            if p.is_file():
                # Skip hidden files and common cache directories
                parts = p.parts
                if any(part.startswith(".") or part == "__pycache__" for part in parts):
                    continue
                yield p


def _search_file(
    file_path: Path,
    pattern: re.Pattern,
    max_results: int,
    results: list[dict],
) -> None:
    """Search a single file for the pattern."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")
        
        for line_num, line in enumerate(lines, 1):
            if len(results) >= max_results:
                return
            
            if pattern.search(line):
                results.append({
                    "file": str(file_path),
                    "line": line_num,
                    "content": line.strip(),
                })
    except (IOError, OSError, UnicodeDecodeError):
        # Skip files that can't be read
        pass


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search for the given pattern.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in (default: current directory)
        file_pattern: Glob pattern for files to include (e.g., '*.py')
        case_sensitive: Whether search is case sensitive
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results
    """
    try:
        # Default to current directory if not specified
        search_path = Path(path) if path else Path.cwd()
        
        if not search_path.is_absolute():
            search_path = search_path.resolve()
        
        # Check allowed root
        if not _is_allowed(str(search_path)):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"
        
        # Collect results
        results: list[dict] = []
        
        for file_path in _find_files(search_path, file_pattern):
            if not _is_allowed(str(file_path)):
                continue
            _search_file(file_path, compiled_pattern, max_results, results)
            if len(results) >= max_results:
                break
        
        # Format output
        if not results:
            return f"No matches found for pattern '{pattern}'"
        
        lines = [f"Found {len(results)} match(es) for pattern '{pattern}':", ""]
        for r in results:
            lines.append(f"{r['file']}:{r['line']}: {r['content']}")
        
        if len(results) >= max_results:
            lines.append(f"\n(Results truncated to {max_results} matches)")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error: {e}"
