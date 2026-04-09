"""
Search tool: search for patterns in files and directories.

Provides grep-like functionality for finding code patterns,
helping the meta agent analyze and understand the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for search operations."""
    return {
        "name": "search",
        "description": "Search for patterns in files and directories. Supports regex and text search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal text)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', default: all files)",
                },
                "regex": {
                    "type": "boolean",
                    "description": "Whether to treat pattern as regex (default: True)",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive (default: False)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
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
    path: str = ".",
    file_pattern: str = "*",
    regex: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute search operation.

    Args:
        pattern: Search pattern
        path: Directory or file to search in
        file_pattern: File pattern to match
        regex: Whether to treat pattern as regex
        case_sensitive: Whether search is case sensitive
        max_results: Maximum number of results

    Returns:
        Search results as formatted string
    """
    # Validate inputs
    if not isinstance(pattern, str):
        return f"Error: pattern must be a string, got {type(pattern).__name__}"
    if not pattern:
        return "Error: pattern cannot be empty"
    
    if not isinstance(path, str):
        return f"Error: path must be a string, got {type(path).__name__}"
    
    try:
        p = Path(path)
    except Exception as e:
        return f"Error: invalid path '{path}': {e}"

    # Scope check
    if _ALLOWED_ROOT is not None:
        try:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        except Exception as e:
            return f"Error: path resolution failed: {e}"

    # Validate max_results
    if not isinstance(max_results, int):
        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            max_results = 50
    max_results = max(1, min(200, max_results))

    results = []
    count = 0

    try:
        if p.is_file():
            # Search in single file
            files_to_search = [p]
        elif p.is_dir():
            # Find files matching pattern
            if file_pattern and file_pattern != "*":
                # Use glob for file pattern matching
                files_to_search = list(p.rglob(file_pattern))
                files_to_search = [f for f in files_to_search if f.is_file()]
            else:
                # Search all files
                files_to_search = [f for f in p.rglob("*") if f.is_file()]
        else:
            return f"Error: path '{path}' does not exist"

        # Compile regex if needed
        if regex:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled_pattern = re.compile(pattern, flags)
            except re.error as e:
                return f"Error: invalid regex pattern: {e}"
        else:
            # Literal text search
            if not case_sensitive:
                pattern = pattern.lower()

        # Search in files
        for file_path in files_to_search:
            if count >= max_results:
                break

            # Skip binary files and very large files
            try:
                stat = file_path.stat()
                if stat.st_size > 10 * 1024 * 1024:  # Skip files > 10MB
                    continue
            except:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        if count >= max_results:
                            break

                        line_to_search = line if case_sensitive else line.lower()

                        if regex:
                            match = compiled_pattern.search(line_to_search)
                        else:
                            match = pattern in line_to_search

                        if match:
                            # Format the match
                            match_text = line.strip()
                            if len(match_text) > 200:
                                match_text = match_text[:200] + "..."
                            rel_path = file_path.relative_to(p) if p.is_dir() else file_path.name
                            results.append(f"{rel_path}:{line_num}: {match_text}")
                            count += 1

            except (IOError, OSError, UnicodeDecodeError):
                # Skip files that can't be read
                continue
            except Exception as e:
                # Log but continue on other errors
                continue

    except Exception as e:
        return f"Error during search: {type(e).__name__}: {e}"

    if not results:
        return f"No matches found for pattern '{pattern}'"

    header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
    if count >= max_results:
        header = f"Found {len(results)}+ match(es) for pattern '{pattern}' (showing first {max_results}):\n"

    return header + "\n".join(results)
