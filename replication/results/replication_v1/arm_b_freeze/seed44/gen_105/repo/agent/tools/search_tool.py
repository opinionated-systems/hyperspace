"""
Search tool: find patterns in files using grep/ripgrep.

Provides file search capabilities to help locate code patterns,
function definitions, and references across the codebase.
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
            "Useful for finding function definitions, variable usage, etc."
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


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[:max_len] + f"\n<... truncated {len(content) - max_len} chars ...>"
    return content


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
) -> str:
    """Search for pattern in files. Returns matching lines."""
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

        # Build grep command
        cmd = ["grep", "-rn", "-E", pattern]
        
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add path
        cmd.append(str(p))

        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Found matches
            return _truncate(result.stdout, 8000)
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            # Error
            return f"Search error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: search timed out (max 30s)"
    except Exception as e:
        return f"Error: {e}"
