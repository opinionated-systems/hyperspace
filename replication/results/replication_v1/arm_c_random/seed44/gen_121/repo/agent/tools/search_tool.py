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
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive (default: true).",
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
    """Search for pattern in files with enhanced error handling."""
    try:
        # Validate inputs
        if not pattern:
            return "Error: pattern is required."
        if not path:
            return "Error: path is required."
        
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path. Use the full path starting with /"

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Path {path} is outside allowed root {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist. Check the path and try again."

        # Build grep command with proper options
        grep_opts = ["-rn"] if p.is_dir() else ["-n"]
        grep_opts.append("-I")  # Ignore binary files
        grep_opts.append("-E")  # Use extended regex for better pattern matching
        
        # Add case-insensitive flag if requested
        if not case_sensitive:
            grep_opts.append("-i")
        
        # Escape the pattern to handle special characters safely
        # Use single quotes to prevent shell expansion
        if p.is_dir():
            cmd = ["grep"] + grep_opts + ["--include", file_extension or "*"]
            cmd.append(pattern)
            cmd.append(str(p))
        else:
            cmd = ["grep"] + grep_opts + [pattern, str(p)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_ALLOWED_ROOT,
            timeout=30,  # Add timeout to prevent hanging on large directories
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            total_matches = len(lines)
            if total_matches > 50:
                return f"Found {total_matches} matches (showing first 50):\n" + "\n".join(lines[:50]) + f"\n... ({total_matches - 50} more matches truncated)"
            return f"Found {total_matches} matches:\n" + result.stdout
        elif result.returncode == 1:
            return "No matches found. Try a different pattern or check your spelling."
        else:
            error_msg = result.stderr.strip()
            if "Permission denied" in error_msg:
                return f"Error: Permission denied while searching. {error_msg}"
            return f"Error during search: {error_msg}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try searching a smaller directory or using a more specific pattern."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
