"""
Search tool: search for patterns in files using grep.

Provides file content search capabilities to help agents explore codebases.
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
            "Returns matching lines with line numbers."
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
            return f"Error: {p} does not exist."

        # Build grep command
        cmd = ["grep", "-rn", "--include"]
        if file_extension:
            cmd.append(f"*{file_extension}")
        else:
            cmd.append("*")
        cmd.extend([pattern, str(p)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                # Truncate if too many matches
                truncated = "\n".join(lines[:25]) + f"\n... ({len(lines) - 50} more matches) ...\n" + "\n".join(lines[-25:])
                return f"Found {len(lines)} matches:\n{truncated}"
            return f"Found {len(lines)} matches:\n{result.stdout}"
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}' in {p}"
        else:
            return f"Error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
