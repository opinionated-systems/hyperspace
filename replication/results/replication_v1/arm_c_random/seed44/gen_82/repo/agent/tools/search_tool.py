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
        # Validate inputs
        if not pattern or not pattern.strip():
            return "Error: Empty search pattern provided"
        
        if not path or not path.strip():
            return "Error: Empty path provided"
        
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

        # Validate pattern to avoid regex errors
        try:
            import re
            re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}. Try escaping special characters like . or *"

        # Build grep command
        # Use -E for extended regex, -n for line numbers, -r for recursive
        if p.is_dir():
            cmd = ["grep", "-rnE", "--include", file_extension or "*", pattern, str(p)]
        else:
            cmd = ["grep", "-nE", pattern, str(p)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_ALLOWED_ROOT,
            timeout=30,  # Add timeout to prevent hanging on large directories
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Filter out empty lines
            lines = [line for line in lines if line.strip()]
            
            if not lines:
                return "No matches found."
            
            if len(lines) > 50:
                return "Matches (first 50):\n" + "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches truncated)"
            return f"Matches ({len(lines)} found):\n" + "\n".join(lines)
        elif result.returncode == 1:
            return "No matches found."
        else:
            error_msg = result.stderr.strip()
            if "Permission denied" in error_msg:
                return f"Error: Permission denied accessing some files. {error_msg}"
            return f"Error: {error_msg}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30s. Try a more specific pattern or narrower scope."
    except Exception as e:
        return f"Error: {e}"
