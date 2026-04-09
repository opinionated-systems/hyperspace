"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns
within the codebase. Useful for finding code references.
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
            "Search for patterns in files using grep. "
            "Returns matching lines with file paths and line numbers. "
            "Useful for finding code references, function definitions, etc."
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
            return f"Error: {p} does not exist."

        # Build search command
        if p.is_dir():
            # Use grep -r for recursive search
            # --include expects a glob pattern like "*.py", not just ".py"
            if file_extension:
                include_pattern = f"*{file_extension}" if not file_extension.startswith("*") else file_extension
            else:
                include_pattern = "*"
            cmd = ["grep", "-rn", "--include", include_pattern, pattern, str(p)]
        else:
            # Single file search
            cmd = ["grep", "-n", pattern, str(p)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                # Truncate if too many results
                return (
                    f"Found {len(lines)} matches (showing first 50):\n"
                    + "\n".join(lines[:50])
                    + f"\n... and {len(lines) - 50} more matches"
                )
            return f"Found {len(lines)} matches:\n" + result.stdout
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}'"
        else:
            return f"Search error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
