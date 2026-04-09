"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities
to help the meta agent locate code patterns efficiently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content. "
            "Commands: grep (search content), find (find files by name). "
            "Useful for locating code patterns across the codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for grep, glob for find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
            },
            "required": ["command", "pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_scope(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True, ""
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate(content: str, max_len: int = 8000) -> str:
    """Truncate long output."""
    if len(content) > max_len:
        lines = content.split("\n")
        if len(lines) > 100:
            return "\n".join(lines[:50]) + f"\n... ({len(lines) - 100} lines omitted) ...\n" + "\n".join(lines[-50:])
        return content[:max_len // 2] + "\n<output clipped>\n" + content[-max_len // 2:]
    return content


def tool_function(
    command: str,
    pattern: str,
    path: str,
    file_extension: str | None = None,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        ok, msg = _check_scope(str(p))
        if not ok:
            return msg

        if command == "grep":
            return _grep(pattern, str(p), file_extension)
        elif command == "find":
            return _find(pattern, str(p))
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, path: str, file_extension: str | None = None) -> str:
    """Search for pattern in files under path."""
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory."

    # Build grep command
    cmd = ["grep", "-r", "-n", "-I", "--include", file_extension or "*.py", pattern, path]
    if file_extension:
        cmd = ["grep", "-r", "-n", "-I", "--include", f"*{file_extension}", pattern, path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout if result.stdout else result.stderr

    if not output.strip():
        return f"No matches found for '{pattern}' in {path}"

    return _truncate(output)


def _find(pattern: str, path: str) -> str:
    """Find files matching pattern under path."""
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory."

    # Use find command with case-insensitive name matching
    cmd = ["find", path, "-type", "f", "-iname", pattern, "-not", "-path", "*/\.*"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    if not output.strip():
        return f"No files matching '{pattern}' found in {path}"

    return _truncate(output, 5000)
