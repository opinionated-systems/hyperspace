"""
Search tool: find files and search for patterns in the codebase.

Provides grep-like functionality and file finding capabilities.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and patterns in the codebase. "
            "Commands: grep (search text), find (find files by name). "
            "Results are truncated to avoid overwhelming output."
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
                    "description": "Pattern to search for (grep) or filename pattern (find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to limit search (e.g., '*.py').",
                },
            },
            "required": ["command", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate(content: str, max_len: int = 5000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n... [output truncated] ...\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
) -> str:
    """Execute a search command."""
    try:
        search_path = path or "."
        p = Path(search_path)

        if not p.is_absolute():
            p = Path(os.getcwd()) / p

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if command == "grep":
            return _grep(pattern, str(p), file_pattern)
        elif command == "find":
            return _find(pattern, str(p))
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, path: str, file_pattern: str | None = None) -> str:
    """Search for pattern in files using grep."""
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"

    # Build grep command
    cmd = ["grep", "-r", "-n", "-I", "--exclude-dir=.*"]
    
    if file_pattern:
        cmd.extend(["--include", file_pattern])
    
    cmd.extend([pattern, path])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            return f"Found matches for '{pattern}':\n{_truncate(result.stdout, 5000)}"
        elif result.returncode == 1:
            return f"No matches found for '{pattern}' in {path}"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: grep timed out after 30s"
    except Exception as e:
        return f"Error running grep: {e}"


def _find(pattern: str, path: str) -> str:
    """Find files by name pattern."""
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"

    try:
        # Use find command with case-insensitive name matching
        result = subprocess.run(
            ["find", path, "-type", "f", "-iname", f"*{pattern}*", "-not", "-path", "*/\.*"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            if result.stdout.strip():
                return f"Files matching '{pattern}':\n{_truncate(result.stdout, 5000)}"
            else:
                return f"No files found matching '{pattern}' in {path}"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: find timed out after 30s"
    except Exception as e:
        return f"Error running find: {e}"
