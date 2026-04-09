"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality and file finding capabilities
to help the meta agent navigate and understand the codebase.
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
            "Commands: find_files (glob patterns), grep (content search), "
            "find_function (find function definitions)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_function"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (glob for find_files, regex for grep).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Filter by file extension (e.g., '.py').",
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
    """Get the search path, scoped to allowed root."""
    if path is None:
        return _ALLOWED_ROOT or os.getcwd()
    
    resolved = os.path.abspath(path)
    if _ALLOWED_ROOT is not None:
        if not resolved.startswith(_ALLOWED_ROOT):
            return _ALLOWED_ROOT
    return resolved


def _truncate_output(output: str, max_len: int = 10000) -> str:
    """Truncate output to prevent context overflow."""
    if len(output) > max_len:
        lines = output.split('\n')
        if len(lines) > 100:
            return '\n'.join(lines[:50]) + f"\n... ({len(lines) - 100} lines omitted) ...\n" + '\n'.join(lines[-50:])
        return output[:max_len//2] + "\n... [output truncated] ...\n" + output[-max_len//2:]
    return output


def _find_files(pattern: str, path: str, max_results: int = 50) -> str:
    """Find files matching a glob pattern."""
    try:
        result = subprocess.run(
            ["find", path, "-type", "f", "-name", pattern],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        
        files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        files = [f for f in files if f]  # Remove empty strings
        
        if len(files) > max_results:
            files = files[:max_results]
            truncated_msg = f"\n... (truncated to {max_results} results) ..."
        else:
            truncated_msg = ""
        
        return f"Found {len(files)} files matching '{pattern}':\n" + '\n'.join(files) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, path: str, file_extension: str | None = None, max_results: int = 50) -> str:
    """Search for content matching a regex pattern."""
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n", "-E", pattern]
        
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add path
        cmd.append(path)
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in [0, 1]:
            return f"Error: {result.stderr}"
        
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        lines = [l for l in lines if l]  # Remove empty strings
        
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (truncated to {max_results} results) ..."
        else:
            truncated_msg = ""
        
        return f"Found {len(lines)} matches for '{pattern}':\n" + '\n'.join(lines) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"


def _find_function(function_name: str, path: str, max_results: int = 50) -> str:
    """Find function or class definitions."""
    try:
        # Search for Python function/class definitions
        pattern = f"^(def|class)\\s+{function_name}\\b"
        return _grep(pattern, path, file_extension=".py", max_results=max_results)
    except Exception as e:
        return f"Error: {e}"


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        search_path = _get_search_path(path)
        
        if command == "find_files":
            return _find_files(pattern, search_path, max_results)
        elif command == "grep":
            return _grep(pattern, search_path, file_extension, max_results)
        elif command == "find_function":
            return _find_function(pattern, search_path, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"
