"""
Search tool: find text patterns in files.

Provides grep-like functionality to search for patterns across the codebase.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files using grep. "
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
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search.",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 50,
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
    case_sensitive: bool = False,
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
            return f"Error: {path} does not exist."

        # Build grep command
        cmd = ["grep"]
        
        if not case_sensitive:
            cmd.append("-i")  # Case insensitive
        
        if recursive and p.is_dir():
            cmd.append("-r")
        
        cmd.extend(["-n", "--include=*.py", "--include=*.txt", "--include=*.md"])
        cmd.append(pattern)
        cmd.append(str(p))

        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 2:
            return f"Error: grep failed - {result.stderr}"
        
        if result.returncode == 1 or not result.stdout:
            return f"No matches found for '{pattern}' in {path}"

        # Truncate results if too many
        lines = result.stdout.strip().split("\n")
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more results)")

        return f"Found {len(lines)} match(es) for '{pattern}':\n" + "\n".join(lines)

    except subprocess.TimeoutExpired:
        return f"Error: search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"
