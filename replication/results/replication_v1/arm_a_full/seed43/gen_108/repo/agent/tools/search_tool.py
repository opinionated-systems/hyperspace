"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns within files.
Useful for finding code references, TODOs, or specific content.
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
            "Search for patterns in files using grep-like functionality. "
            "Can search within a directory or specific file. "
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
                    "description": "Directory or file to search in. Absolute path.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive (default: True).",
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
            return f"Error: {p} does not exist."

        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE

        if p.is_file():
            files = [p]
        else:
            # Find all files, optionally filtered by extension
            if file_extension:
                files = list(p.rglob(f"*{file_extension}"))
            else:
                files = [f for f in p.rglob("*") if f.is_file()]

        for file_path in files:
            try:
                content = file_path.read_text(errors="ignore")
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, flags):
                        matches.append(f"{file_path}:{i}:{line.strip()}")
            except Exception:
                continue

        if not matches:
            return f"No matches found for pattern '{pattern}' in {path}"

        # Limit output
        if len(matches) > 100:
            return (
                f"Found {len(matches)} matches for '{pattern}':\n"
                + "\n".join(matches[:50])
                + f"\n... ({len(matches) - 50} more matches) ..."
            )

        return f"Found {len(matches)} matches for '{pattern}':\n" + "\n".join(matches)

    except Exception as e:
        return f"Error: {e}"
