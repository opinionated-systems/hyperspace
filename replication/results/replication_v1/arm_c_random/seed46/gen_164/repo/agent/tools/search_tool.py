"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
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
            "Supports regex patterns and can search specific files or directories. "
            "Returns matching lines with line numbers. "
            "Case-insensitive by default."
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
                    "description": "Absolute path to file or directory to search.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
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


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
) -> str:
    """Execute a search using grep."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {p} does not exist."

        # Build grep command with case-insensitive flag
        cmd = ["grep", "-rni", "--include", file_extension or "*", pattern, str(p)]
        
        # If searching a single file, adjust command
        if p.is_file():
            cmd = ["grep", "-ni", pattern, str(p)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return _truncate(result.stdout, 5000)
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            return f"Error: {result.stderr}"

    except Exception as e:
        return f"Error: {e}"
