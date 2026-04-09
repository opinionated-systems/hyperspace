"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

# Configuration constants
DEFAULT_MAX_RESULTS = 50  # Default maximum number of results to return
GREP_TIMEOUT = 30  # Timeout for grep operations in seconds
MAX_RESULTS_LIMIT = 1000  # Hard limit for max_results to prevent abuse


def tool_info() -> dict:
    """Return tool metadata for the search tool."""
    return {
        "name": "search",
        "description": (
            "Search for files and content. "
            "Commands: find_files (glob patterns), grep (search content), "
            "find_in_files (search multiple files)."
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
                    "description": "Directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (glob for find_files, regex for grep).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to limit search (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Maximum number of results to return (default {DEFAULT_MAX_RESULTS}, max {MAX_RESULTS_LIMIT}).",
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


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_output(lines: list[str], max_lines: int = 50) -> str:
    """Truncate output to max_lines."""
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more results)"
    return "\n".join(lines)


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Execute a search command.
    
    Args:
        command: The search command to run (find_files, grep, find_in_files).
        path: Directory to search in (absolute path).
        pattern: Search pattern (glob for find_files, regex for grep).
        file_pattern: Optional file pattern to limit search (e.g., '*.py').
        max_results: Maximum number of results to return.
        
    Returns:
        Search results or error message.
    """
    try:
        # Validate max_results
        if not isinstance(max_results, int) or max_results < 1:
            max_results = DEFAULT_MAX_RESULTS
        max_results = min(max_results, MAX_RESULTS_LIMIT)
        
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if command == "find_files":
            return _find_files(p, pattern, max_results)
        elif command == "grep":
            return _grep(p, pattern, file_pattern, max_results)
        elif command == "find_in_files":
            return _find_in_files(p, pattern, file_pattern, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(path: Path, pattern: str, max_results: int) -> str:
    """Find files matching glob pattern."""
    try:
        matches = list(path.rglob(pattern))
        # Filter to files only
        files = [str(m.relative_to(path)) for m in matches if m.is_file()]
        files.sort()
        
        if not files:
            return f"No files matching '{pattern}' found in {path}"
        
        output = _truncate_output(files, max_results)
        return f"Found {len(files)} files matching '{pattern}':\n{output}"
    except Exception as e:
        return f"Error finding files: {e}"


def _grep(path: Path, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Search for pattern in files using grep.
    
    Args:
        path: Directory to search in.
        pattern: Regex pattern to search for.
        file_pattern: Optional file pattern to limit search.
        max_results: Maximum number of results to return.
        
    Returns:
        Search results or error message.
    """
    try:
        # Validate pattern
        if not pattern or not pattern.strip():
            return "Error: Empty search pattern provided."
        
        # Validate regex pattern by compiling it
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}. Please use a valid regular expression."
        
        cmd = ["grep", "-r", "-n", "-I", "--include"]
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        cmd.extend(["-e", pattern, str(path)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=GREP_TIMEOUT,
        )
        
        if result.returncode == 1:
            return f"No matches for '{pattern}' in {path}"
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        
        lines = result.stdout.strip().split("\n")
        if not lines or lines == [""]:
            return f"No matches for '{pattern}' in {path}"
        
        output = _truncate_output(lines, max_results)
        return f"Found {len(lines)} matches for '{pattern}':\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after {GREP_TIMEOUT}s"
    except Exception as e:
        return f"Error searching: {e}"


def _find_in_files(path: Path, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Find files containing pattern.
    
    Args:
        path: Directory to search in.
        pattern: Regex pattern to search for.
        file_pattern: Optional file pattern to limit search.
        max_results: Maximum number of results to return.
        
    Returns:
        List of files containing the pattern, or error message.
    """
    try:
        # Validate pattern
        if not pattern or not pattern.strip():
            return "Error: Empty search pattern provided."
        
        # Validate regex pattern by compiling it
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}. Please use a valid regular expression."
        
        cmd = ["grep", "-r", "-l", "-I", "--include"]
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        cmd.extend(["-e", pattern, str(path)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=GREP_TIMEOUT,
        )
        
        if result.returncode == 1:
            return f"No files containing '{pattern}' in {path}"
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        
        files = result.stdout.strip().split("\n")
        files = [f for f in files if f]  # Remove empty strings
        files.sort()
        
        if not files:
            return f"No files containing '{pattern}' in {path}"
        
        output = _truncate_output(files, max_results)
        return f"Found {len(files)} files containing '{pattern}':\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after {GREP_TIMEOUT}s"
    except Exception as e:
        return f"Error searching: {e}"
