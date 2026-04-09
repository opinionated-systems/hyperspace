"""
File search tool: search for patterns in files using grep-like functionality.

Provides content-based search across files, complementing the path-based search tool.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_search",
        "description": (
            "Search for text patterns in files. "
            "Uses grep-like functionality to find matches. "
            "Returns matching lines with file paths and line numbers. "
            "Supports regex patterns. "
            "Avoid searching in very large directories."
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
                    "description": "Absolute path to directory or file to search in.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive (default: True).",
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


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for pattern in files."""
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
            return f"Error: {path} does not exist."

        results = []
        count = 0

        # Build file list
        if p.is_file():
            files = [p]
        else:
            if file_extension:
                files = list(p.rglob(f"*{file_extension}"))
            else:
                files = list(p.rglob("*"))
            # Filter to files only and exclude hidden directories
            files = [f for f in files if f.is_file() and not any(part.startswith(".") for part in f.parts)]

        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        # Search files
        for file_path in files:
            if count >= max_results:
                break

            try:
                # Skip binary files and very large files
                if file_path.stat().st_size > 10_000_000:  # 10MB
                    continue

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Truncate long lines
                            if len(line) > 200:
                                line = line[:100] + "..." + line[-100:]
                            results.append(f"{file_path}:{line_num}: {line.rstrip()}")
                            count += 1
                            if count >= max_results:
                                break
            except (IOError, OSError, UnicodeDecodeError):
                continue

        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"

        header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
        return header + "\n".join(results)

    except Exception as e:
        return f"Error: {e}"
