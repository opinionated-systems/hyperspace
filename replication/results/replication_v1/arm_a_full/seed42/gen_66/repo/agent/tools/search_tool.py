"""
Search tool: search for files and content within the codebase.

Provides grep-like search and file finding capabilities to help
agents locate code patterns and files efficiently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search tool for finding files and content. "
            "Commands: grep (search file contents), find (search for files by name)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to search within (directory).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for grep, glob for find).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
            },
            "required": ["command", "path", "pattern"],
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
    command: str,
    path: str,
    pattern: str,
    file_extension: str | None = None,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {path} does not exist."

        if not p.is_dir():
            return f"Error: {path} is not a directory."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if command == "grep":
            return _grep(p, pattern, file_extension)
        elif command == "find":
            return _find(p, pattern)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(directory: Path, pattern: str, file_extension: str | None = None) -> str:
    """Search file contents using grep."""
    cmd = ["grep", "-r", "-n", "-I", "--include", file_extension or "*", pattern, str(directory)]
    if file_extension:
        cmd[5] = f"*{file_extension}"
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode == 0:
        return _truncate(f"Matches found:\n{result.stdout}", 8000)
    elif result.returncode == 1:
        return f"No matches found for pattern '{pattern}' in {directory}"
    else:
        return f"Error running grep: {result.stderr}"


def _find(directory: Path, pattern: str) -> str:
    """Search for files by name pattern."""
    cmd = ["find", str(directory), "-type", "f", "-name", pattern, "-not", "-path", "*/\.*"]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode == 0:
        files = result.stdout.strip()
        if files:
            return _truncate(f"Files found:\n{files}", 8000)
        return f"No files matching '{pattern}' found in {directory}"
    else:
        return f"Error running find: {result.stderr}"
