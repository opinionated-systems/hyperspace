"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns
within the allowed root directory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Searches recursively within the allowed root directory. "
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
                    "description": "Directory to search in (absolute path, defaults to allowed root).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for pattern in files."""
    try:
        # Determine search directory
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: no path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)

        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if not os.path.isdir(search_path):
            return f"Error: {search_path} is not a directory."

        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include", file_extension or "*", pattern, search_path]
        
        # If file extension specified, adjust include pattern
        if file_extension:
            if not file_extension.startswith("."):
                file_extension = "." + file_extension
            cmd = ["grep", "-r", "-n", "-I", "--include", f"*{file_extension}", pattern, search_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more results)")
            return "Search results:\n" + "\n".join(lines)
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}'"
        else:
            return f"Error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"
