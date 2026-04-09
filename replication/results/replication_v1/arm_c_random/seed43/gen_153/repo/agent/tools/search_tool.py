"""
Search tool: find files and search for content within files.

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
            "Search for files and content within files. "
            "Commands: find_files (by name pattern), grep (search content), "
            "and find_in_files (search text in file contents)."
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
                    "description": "File name pattern for find_files (e.g., '*.py').",
                },
                "query": {
                    "type": "string",
                    "description": "Search text for grep/find_in_files.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to limit search (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
            },
            "required": ["command", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path_allowed(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True, ""
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_output(output: str, max_lines: int = 100) -> str:
    """Truncate output to max lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
    query: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error_msg = _check_path_allowed(str(p))
        if not allowed:
            return error_msg
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        if command == "find_files":
            return _find_files(p, pattern or "*", max_results)
        elif command == "grep":
            return _grep(p, query or "", file_pattern, max_results)
        elif command == "find_in_files":
            return _find_in_files(p, query or "", file_pattern, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(directory: Path, pattern: str, max_results: int) -> str:
    """Find files matching a name pattern."""
    try:
        cmd = ["find", str(directory), "-type", "f", "-name", pattern]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return f"Error: find command failed: {result.stderr}"
        
        files = [f for f in result.stdout.strip().split("\n") if f]
        if not files:
            return f"No files found matching '{pattern}' in {directory}"
        
        # Limit results
        if len(files) > max_results:
            files = files[:max_results]
            truncated_msg = f"\n... (showing {max_results} of {len(files)} results)"
        else:
            truncated_msg = ""
        
        return f"Found {len(files)} files matching '{pattern}':\n" + "\n".join(files) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


def _grep(directory: Path, query: str, file_pattern: str | None, max_results: int) -> str:
    """Search for text in file contents using grep."""
    try:
        if not query:
            return "Error: query is required for grep"
        
        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include"]
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        cmd.extend([query, str(directory)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in [0, 1]:
            return f"Error: grep failed: {result.stderr}"
        
        lines = [l for l in result.stdout.strip().split("\n") if l]
        if not lines:
            return f"No matches found for '{query}' in {directory}"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (showing {max_results} of {len(lines)} matches)"
        else:
            truncated_msg = ""
        
        return f"Found {len(lines)} matches for '{query}':\n" + "\n".join(lines) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


def _find_in_files(directory: Path, query: str, file_pattern: str | None, max_results: int) -> str:
    """Search for text in file contents with context."""
    try:
        if not query:
            return "Error: query is required for find_in_files"
        
        # Use grep with context lines
        cmd = ["grep", "-r", "-n", "-I", "-C", "2", "--include"]
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        cmd.extend([query, str(directory)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode not in [0, 1]:
            return f"Error: grep failed: {result.stderr}"
        
        output = result.stdout.strip()
        if not output:
            return f"No matches found for '{query}' in {directory}"
        
        lines = output.split("\n")
        if len(lines) > max_results * 5:  # Account for context lines
            lines = lines[:max_results * 5]
            truncated_msg = f"\n... (output truncated)"
        else:
            truncated_msg = ""
        
        return f"Search results for '{query}':\n" + "\n".join(lines) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"
