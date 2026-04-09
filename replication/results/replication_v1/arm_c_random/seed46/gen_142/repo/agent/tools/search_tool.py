"""
Search tool: find files and search for content within files.

Provides grep-like functionality and file finding capabilities
to help navigate and understand codebases efficiently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content within files. "
            "Commands: find_files (by name pattern), grep (search content), "
            "find_in_files (search multiple files). "
            "Useful for navigating and understanding codebases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (absolute).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename pattern for find_files, regex for grep).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to limit grep search (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                    "default": 50,
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


def _truncate_output(output: str, max_lines: int = 100) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def _check_path_allowed(path: str) -> str | None:
    """Check if path is within allowed root. Returns error message if not allowed."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return None


def _find_files(path: str, pattern: str, max_results: int) -> str:
    """Find files matching a name pattern."""
    error = _check_path_allowed(path)
    if error:
        return error

    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"

    try:
        # Use find command for efficient file searching
        cmd = [
            "find", path, "-type", "f", "-name", pattern,
            "-not", "-path", "*/\.*",  # Exclude hidden files
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return f"Error running find: {result.stderr}"
        
        files = [f for f in result.stdout.strip().split("\n") if f]
        if not files:
            return f"No files matching '{pattern}' found in {path}"
        
        # Truncate if too many results
        if len(files) > max_results:
            files = files[:max_results]
            truncated_msg = f"\n... (showing {max_results} of {len(files)}+ matches)"
        else:
            truncated_msg = ""
        
        return f"Found {len(files)} file(s) matching '{pattern}':\n" + "\n".join(files) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: search timed out (took longer than 30 seconds)"
    except Exception as e:
        return f"Error: {e}"


def _grep(path: str, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Search for content within files using grep."""
    error = _check_path_allowed(path)
    if error:
        return error

    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"

    try:
        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", pattern, path]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in (0, 1):
            return f"Error running grep: {result.stderr}"
        
        lines = [l for l in result.stdout.strip().split("\n") if l]
        if not lines:
            return f"No matches for '{pattern}' in {path}"
        
        # Truncate if too many results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (showing {max_results} of {len(lines)}+ matches)"
        else:
            truncated_msg = ""
        
        return f"Found {len(lines)} match(es) for '{pattern}':\n" + "\n".join(lines) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: search timed out (took longer than 30 seconds)"
    except Exception as e:
        return f"Error: {e}"


def _find_in_files(path: str, pattern: str, max_results: int) -> str:
    """Search for pattern in specific file types (Python, JS, etc.)."""
    error = _check_path_allowed(path)
    if error:
        return error

    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"

    try:
        # Search in common code files
        cmd = [
            "grep", "-r", "-n", "-I",
            "--include", "*.py",
            "--include", "*.js",
            "--include", "*.ts",
            "--include", "*.java",
            "--include", "*.cpp",
            "--include", "*.c",
            "--include", "*.h",
            "--include", "*.md",
            "--include", "*.txt",
            "--include", "*.json",
            "--include", "*.yaml",
            "--include", "*.yml",
            pattern, path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode not in (0, 1):
            return f"Error running search: {result.stderr}"
        
        lines = [l for l in result.stdout.strip().split("\n") if l]
        if not lines:
            return f"No matches for '{pattern}' in code files under {path}"
        
        # Truncate if too many results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (showing {max_results} of {len(lines)}+ matches)"
        else:
            truncated_msg = ""
        
        return f"Found {len(lines)} match(es) in code files:\n" + "\n".join(lines) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: search timed out (took longer than 30 seconds)"
    except Exception as e:
        return f"Error: {e}"


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    if not os.path.isabs(path):
        return f"Error: {path} is not an absolute path"

    if command == "find_files":
        return _find_files(path, pattern, max_results)
    elif command == "grep":
        return _grep(path, pattern, file_pattern, max_results)
    elif command == "find_in_files":
        return _find_in_files(path, pattern, max_results)
    else:
        return f"Error: unknown command {command}"
