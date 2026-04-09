"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns
within files in the repository.
"""

from __future__ import annotations

import os
import re
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
                    "description": "Directory or file to search in (absolute path).",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories.",
                    "default": True,
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
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
    recursive: bool = True,
    file_extension: str | None = None,
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

        # Compile regex
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"

        results = []
        max_results = 100

        if p.is_file():
            files = [p]
        else:
            if recursive:
                files = list(p.rglob("*"))
            else:
                files = list(p.iterdir())
            files = [f for f in files if f.is_file()]

        # Filter by extension if specified
        if file_extension:
            files = [f for f in files if f.suffix == file_extension]

        # Skip hidden files and binary files
        files = [
            f for f in files
            if not any(part.startswith(".") for part in f.parts)
            and not f.name.endswith(('.pyc', '.so', '.dll', '.exe'))
        ]

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        results.append(f"{file_path}:{line_num}: {line[:200]}")
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
            except Exception:
                continue

        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"

        header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
        if len(results) >= max_results:
            header = f"Found 100+ matches for pattern '{pattern}' (showing first 100):\n"

        return header + "\n".join(results)

    except Exception as e:
        return f"Error: {e}"
