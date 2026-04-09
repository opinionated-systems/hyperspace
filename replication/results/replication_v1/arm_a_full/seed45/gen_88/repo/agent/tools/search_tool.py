"""
Search tool: search for patterns in files using grep/find.

Provides file search capabilities for the meta agent to find
code patterns, function definitions, and references.
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
            "Can search file contents or find files by name. "
            "Results are truncated if too large."
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
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to search (e.g., '*.py'). Optional.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Search in file contents or search for filenames.",
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
    file_pattern: str | None = None,
    search_type: str = "content",
) -> str:
    """Execute a search command."""
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No search path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            # Validate path is within allowed root
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: Search path '{path}' is outside allowed root '{_ALLOWED_ROOT}'"

        if not os.path.exists(search_path):
            return f"Error: Path '{search_path}' does not exist."

        if search_type == "filename":
            # Search for files by name pattern
            cmd = ["find", search_path, "-type", "f", "-name", pattern]
            if file_pattern:
                # Add additional filter
                pass
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout
            if result.returncode != 0:
                output += f"\n(find stderr: {result.stderr})"
        else:
            # Search in file contents using grep
            if os.path.isfile(search_path):
                # Search single file
                cmd = ["grep", "-n", "-H", pattern, search_path]
            else:
                # Search directory recursively
                cmd = ["grep", "-r", "-n", "-H", pattern, search_path]
                if file_pattern:
                    cmd.extend(["--include", file_pattern])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout
            if result.returncode != 0 and result.returncode != 1:
                # grep returns 1 when no matches found, which is OK
                output += f"\n(grep stderr: {result.stderr})"
            if not output.strip() and result.returncode == 1:
                output = f"No matches found for pattern '{pattern}'"

        # Truncate if too long
        if len(output) > 10000:
            lines = output.split("\n")
            if len(lines) > 200:
                output = "\n".join(lines[:100]) + f"\n... [{len(lines) - 200} more lines] ...\n" + "\n".join(lines[-100:])
        
        return output if output.strip() else f"No results for pattern '{pattern}'"

    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error: {e}"
