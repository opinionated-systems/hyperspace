"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality and file finding capabilities
to help navigate larger codebases efficiently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content within the codebase. "
            "Commands: find_files (by name pattern), grep (search content), "
            "find_in_files (search content with context)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename pattern for find_files, regex for grep).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern filter for grep (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
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


def _get_search_path(path: str | None) -> str:
    """Get the search path, ensuring it's within allowed root."""
    if path is None:
        return _ALLOWED_ROOT or os.getcwd()
    
    p = Path(path)
    if not p.is_absolute():
        p = Path(_ALLOWED_ROOT or os.getcwd()) / p
    
    resolved = os.path.abspath(str(p))
    if _ALLOWED_ROOT is not None:
        if not resolved.startswith(_ALLOWED_ROOT):
            return _ALLOWED_ROOT
    return resolved


def _truncate_output(output: str, max_lines: int = 100) -> str:
    """Truncate output to max lines with indicator."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def _find_files(pattern: str, path: str, max_results: int) -> str:
    """Find files by name pattern."""
    try:
        result = subprocess.run(
            ["find", path, "-type", "f", "-name", pattern],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        
        files = [f for f in result.stdout.strip().split("\n") if f]
        if not files:
            return f"No files matching '{pattern}' found in {path}"
        
        files = files[:max_results]
        output = f"Found {len(files)} file(s) matching '{pattern}':\n" + "\n".join(files)
        if len(files) == max_results:
            output += "\n... (results truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, path: str, file_pattern: str | None, max_results: int) -> str:
    """Search file contents for pattern."""
    try:
        cmd = ["grep", "-r", "-n", "-l", pattern, path]
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        
        # grep -l returns filenames only, exit code 1 means no matches
        if result.returncode not in (0, 1):
            return f"Error: {result.stderr}"
        
        files = [f for f in result.stdout.strip().split("\n") if f]
        if not files:
            return f"No files containing '{pattern}' found in {path}"
        
        files = files[:max_results]
        output = f"Found {len(files)} file(s) containing '{pattern}':\n" + "\n".join(files)
        if len(files) == max_results:
            output += "\n... (results truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


def _find_in_files(pattern: str, path: str, file_pattern: str | None, max_results: int) -> str:
    """Search file contents with context lines."""
    try:
        cmd = ["grep", "-r", "-n", "-C", "2", pattern, path]
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        
        if result.returncode not in (0, 1):
            return f"Error: {result.stderr}"
        
        lines = [l for l in result.stdout.strip().split("\n") if l]
        if not lines:
            return f"No matches for '{pattern}' found in {path}"
        
        lines = lines[:max_results * 5]  # Approximate lines per match
        output = f"Matches for '{pattern}':\n" + _truncate_output("\n".join(lines), max_results * 5)
        return output
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        search_path = _get_search_path(path)
        
        if command == "find_files":
            return _find_files(pattern, search_path, max_results)
        elif command == "grep":
            return _grep(pattern, search_path, file_pattern, max_results)
        elif command == "find_in_files":
            return _find_in_files(pattern, search_path, file_pattern, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"
