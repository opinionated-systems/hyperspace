"""
Search tool: search for patterns in files and directories.

Provides grep-like functionality to search for text patterns across files.
Supports regex patterns and can search recursively through directories.
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
            "Search for patterns in files and directories. "
            "Supports regex patterns and recursive directory search. "
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
                    "description": "Absolute path to file or directory to search in.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in directories (default: True).",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default: False).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_allowed_root(path: str) -> bool:
    """Check if a path is within the allowed root directory."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def _truncate(content: str, max_len: int = 10000) -> str:
    """Truncate content if it exceeds max length."""
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str,
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search for patterns in files.
    
    Args:
        pattern: The regex pattern to search for
        path: Absolute path to file or directory
        recursive: Whether to search recursively (default True)
        case_sensitive: Whether search is case-sensitive (default False)
        max_results: Maximum number of results to return (default 50)
    
    Returns:
        String with search results showing file paths, line numbers, and matching lines
    """
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if not _is_within_allowed_root(str(p)):
            return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {p} does not exist."

        flags = 0 if case_sensitive else re.IGNORECASE
        results = []
        result_count = 0

        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        if p.is_file():
            # Search in single file
            results.extend(_search_file(p, compiled_pattern))
            result_count = len(results)
        elif p.is_dir():
            # Search in directory
            if recursive:
                files_to_search = list(p.rglob("*"))
            else:
                files_to_search = list(p.iterdir())
            
            for file_path in files_to_search:
                if file_path.is_file() and not _is_binary_file(file_path):
                    file_results = _search_file(file_path, compiled_pattern)
                    results.extend(file_results)
                    result_count += len(file_results)
                    if result_count >= max_results:
                        break

        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"

        # Format results
        formatted_results = []
        for i, (file_path, line_num, line_content) in enumerate(results[:max_results]):
            # Truncate very long lines
            if len(line_content) > 200:
                line_content = line_content[:100] + "..." + line_content[-100:]
            formatted_results.append(f"{file_path}:{line_num}: {line_content}")

        output = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
        if len(results) > max_results:
            output += f"(showing first {max_results} results)\n"
        output += "\n".join(formatted_results)
        
        return _truncate(output)

    except Exception as e:
        return f"Error: {e}"


def _is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary (to skip binary files during search)."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\0" in chunk
    except Exception:
        return True


def _search_file(file_path: Path, pattern: re.Pattern) -> list[tuple[str, int, str]]:
    """Search for pattern in a single file.
    
    Returns:
        List of tuples (file_path_str, line_number, line_content)
    """
    results = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                if pattern.search(line):
                    results.append((str(file_path), line_num, line.rstrip("\n")))
    except Exception:
        # Skip files that can't be read
        pass
    return results
