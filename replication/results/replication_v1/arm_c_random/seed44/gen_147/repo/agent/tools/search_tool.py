"""
Search tool: search for patterns in files.

Provides grep-like functionality to find code patterns.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Returns matching lines with file paths and line numbers. "
            "Useful for finding code patterns before editing."
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

        # Build grep command with proper file extension handling
        if p.is_dir():
            if file_extension:
                # Ensure extension starts with a dot
                ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
                cmd = ["grep", "-rn", "--include", f"*{ext}", pattern, str(p)]
            else:
                cmd = ["grep", "-rn", pattern, str(p)]
        else:
            cmd = ["grep", "-n", pattern, str(p)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_ALLOWED_ROOT,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                return "Matches (first 50):\n" + "\n".join(lines[:50]) + "\n... (truncated)"
            return "Matches:\n" + result.stdout
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Error: {result.stderr}"

    except Exception as e:
        return f"Error: {e}"
