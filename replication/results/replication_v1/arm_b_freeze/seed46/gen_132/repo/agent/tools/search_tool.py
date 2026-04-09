"""
Search tool: search for patterns in files using grep and find.

Provides file search capabilities to help the meta agent analyze
the codebase and find relevant files and code patterns.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep and find. "
            "Helps locate files and code patterns in the codebase. "
            "Useful for finding where functions are defined, "
            "where variables are used, or finding files by name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex for grep, glob for find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory).",
                },
                "command": {
                    "type": "string",
                    "enum": ["grep", "find", "grep_file"],
                    "description": "Search command: 'grep' for content, 'find' for files, 'grep_file' for content in specific file.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Specific file to search in (for grep_file command).",
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


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n... [output truncated] ...\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_path: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        search_path = path or "."
        search_path = os.path.abspath(search_path)
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if command == "find":
            return _find_files(search_path, pattern, max_results)
        elif command == "grep":
            return _grep_content(search_path, pattern, max_results)
        elif command == "grep_file":
            if not file_path:
                return "Error: file_path required for grep_file command."
            return _grep_file(file_path, pattern, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(search_path: str, pattern: str, max_results: int) -> str:
    """Find files matching a glob pattern."""
    try:
        # Use find with -name for glob patterns
        cmd = [
            "find", search_path,
            "-type", "f",
            "-name", pattern,
            "-not", "-path", "*/\.*",  # Exclude hidden files
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error running find: {result.stderr}"
        
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        files = [f for f in files if f]  # Remove empty strings
        
        if not files:
            return f"No files found matching '{pattern}' in {search_path}"
        
        # Limit results
        total = len(files)
        if len(files) > max_results:
            files = files[:max_results]
            truncated_msg = f"\n... and {total - max_results} more files (showing first {max_results})"
        else:
            truncated_msg = ""
        
        output = f"Found {total} file(s) matching '{pattern}':\n" + "\n".join(files) + truncated_msg
        return _truncate(output, 15000)
        
    except subprocess.TimeoutExpired:
        return "Error: find command timed out after 30s. Try a more specific pattern."
    except Exception as e:
        return f"Error: {e}"


def _grep_content(search_path: str, pattern: str, max_results: int) -> str:
    """Search for content in files using grep."""
    try:
        # Use grep with line numbers, recursive, excluding binary files
        cmd = [
            "grep", "-r", "-n", "-I", "--include=*.py",
            "--include=*.txt", "--include=*.md", "--include=*.json",
            "--include=*.yaml", "--include=*.yml",
            "-E", pattern,  # Extended regex
            search_path,
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in (0, 1):
            return f"Error running grep: {result.stderr}"
        
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        lines = [l for l in lines if l]  # Remove empty strings
        
        if not lines:
            return f"No matches found for pattern '{pattern}' in {search_path}"
        
        # Limit results
        total = len(lines)
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... and {total - max_results} more matches (showing first {max_results})"
        else:
            truncated_msg = ""
        
        output = f"Found {total} match(es) for '{pattern}':\n" + "\n".join(lines) + truncated_msg
        return _truncate(output, 15000)
        
    except subprocess.TimeoutExpired:
        return "Error: grep command timed out after 30s. Try a more specific pattern."
    except Exception as e:
        return f"Error: {e}"


def _grep_file(file_path: str, pattern: str, max_results: int) -> str:
    """Search for content in a specific file."""
    try:
        p = Path(file_path)
        if not p.is_absolute():
            return f"Error: {file_path} is not an absolute path."
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if p.is_dir():
            return f"Error: {p} is a directory. Use grep command for directories."
        
        # Use grep with line numbers
        cmd = [
            "grep", "-n", "-E",  # Extended regex
            pattern,
            str(p),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in (0, 1):
            return f"Error running grep: {result.stderr}"
        
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        lines = [l for l in lines if l]  # Remove empty strings
        
        if not lines:
            return f"No matches found for pattern '{pattern}' in {p}"
        
        # Limit results
        total = len(lines)
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... and {total - max_results} more matches (showing first {max_results})"
        else:
            truncated_msg = ""
        
        output = f"Found {total} match(es) in {p}:\n" + "\n".join(lines) + truncated_msg
        return _truncate(output, 15000)
        
    except subprocess.TimeoutExpired:
        return "Error: grep command timed out after 10s."
    except Exception as e:
        return f"Error: {e}"
