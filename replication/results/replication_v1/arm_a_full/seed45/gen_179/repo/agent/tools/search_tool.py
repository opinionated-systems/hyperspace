"""
Search tool: search for patterns in files using grep.

Provides a convenient way to find code patterns, function definitions,
and references across the codebase.
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
            "Useful for finding function definitions, variable references, "
            "or any text pattern across the codebase. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (grep-compatible regex).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path). Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: False.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files."""
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            # Scope check
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if not os.path.exists(search_path):
            return f"Error: Path does not exist: {search_path}"

        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add file extension filter if specified
        if file_extension:
            if not file_extension.startswith("."):
                file_extension = "." + file_extension
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Exclude binary files and hidden directories
        cmd.extend(["--binary-files=without-match"])
        cmd.append(search_path)

        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Found matches
            lines = result.stdout.strip().split("\n")
            if len(lines) > 100:
                # Truncate if too many results
                truncated = lines[:50] + ["... (results truncated, showing first 50 of " + str(len(lines)) + " matches) ..."] + lines[-10:]
                return "\n".join(truncated)
            return result.stdout
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {search_path}"
        else:
            # Error
            return f"Error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error: {e}"
