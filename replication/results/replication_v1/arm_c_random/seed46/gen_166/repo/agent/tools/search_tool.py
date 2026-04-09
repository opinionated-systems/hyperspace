"""
Search tool: search for text patterns in files.

Provides grep-like functionality to find text patterns across files.
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
            "Search for text patterns in files. "
            "Supports regex patterns and can search within specific directories. "
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
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
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


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {p} does not exist."

        results = []
        count = 0

        if p.is_file():
            # Search in single file
            files_to_search = [p]
        else:
            # Search in directory
            if file_extension:
                files_to_search = list(p.rglob(f"*{file_extension}"))
            else:
                files_to_search = list(p.rglob("*"))
            # Filter out directories and hidden files
            files_to_search = [
                f for f in files_to_search 
                if f.is_file() and not any(part.startswith(".") for part in f.parts)
            ]

        for file_path in files_to_search:
            if count >= max_results:
                break

            try:
                content = file_path.read_text(errors="ignore")
                lines = content.split("\n")
                
                for i, line in enumerate(lines, 1):
                    if count >= max_results:
                        break
                    
                    try:
                        if re.search(pattern, line):
                            results.append(f"{file_path}:{i}:{line}")
                            count += 1
                    except re.error:
                        # If regex fails, try literal string match
                        if pattern in line:
                            results.append(f"{file_path}:{i}:{line}")
                            count += 1
            except (IOError, OSError):
                continue

        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"

        truncated = "\n".join(results[:max_results])
        if len(results) >= max_results:
            truncated += f"\n... ({len(results)} total matches, showing first {max_results})"
        
        return f"Found {len(results)} match(es) for '{pattern}':\n{truncated}"

    except Exception as e:
        return f"Error: {e}"
