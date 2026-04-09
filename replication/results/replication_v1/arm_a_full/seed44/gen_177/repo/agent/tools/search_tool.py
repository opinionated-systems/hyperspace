"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities
to help the meta-agent locate code patterns and files.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content. Commands: grep, find. "
            "grep searches file contents for patterns. "
            "find locates files by name pattern."
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
                    "description": "Directory to search in (default: allowed root).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to limit search (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case-sensitive (default: True).",
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
    
    resolved = os.path.abspath(path)
    if _ALLOWED_ROOT is not None:
        if not resolved.startswith(_ALLOWED_ROOT):
            return _ALLOWED_ROOT
    return resolved


def _run_grep(pattern: str, path: str, file_pattern: str | None, max_results: int, case_sensitive: bool = True) -> str:
    """Run grep to search file contents."""
    # Validate path exists
    if not os.path.exists(path):
        return f"Error: Search path '{path}' does not exist."
    
    cmd = ["grep", "-r", "-n", "-I"]
    
    # Add case-insensitive flag if needed
    if not case_sensitive:
        cmd.append("-i")
    
    # Handle special characters in pattern by using fixed-strings mode if needed
    # Check if pattern contains regex special characters that might cause issues
    special_chars = set('.^$*+?{}[]|()\\')
    use_fixed = any(c in pattern for c in special_chars)
    
    if use_fixed:
        cmd.append("-F")  # Fixed strings mode (treat pattern as literal string)
    
    cmd.extend(["--include", file_pattern or "*", pattern, path])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n")
        lines = [line for line in lines if line]
        
        if not lines:
            case_info = " (case-insensitive)" if not case_sensitive else ""
            fixed_info = " (literal string search)" if use_fixed else ""
            # Check if there was an error
            if result.returncode != 0 and result.stderr:
                return f"No matches found for pattern '{pattern}'{case_info}{fixed_info} in {path}. Note: grep exited with code {result.returncode}"
            return f"No matches found for pattern '{pattern}'{case_info}{fixed_info} in {path}"
        
        # Limit results
        total_matches = len(lines)
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({total_matches - max_results} more results)")
        
        case_info = " (case-insensitive)" if not case_sensitive else ""
        fixed_info = " (literal string search)" if use_fixed else ""
        return f"Found {total_matches} matches for '{pattern}'{case_info}{fixed_info}:\n" + "\n".join(lines)
    
    except subprocess.TimeoutExpired:
        return f"Error: grep search timed out after 30s. Try a more specific pattern or narrower file pattern."
    except Exception as e:
        return f"Error running grep: {e}"


def _run_find(pattern: str, path: str, max_results: int) -> str:
    """Run find to locate files by name."""
    # Validate path exists
    if not os.path.exists(path):
        return f"Error: Search path '{path}' does not exist."
    
    # Use -maxdepth to prevent extremely deep searches that could be slow
    cmd = ["find", path, "-maxdepth", "10", "-name", pattern, "-type", "f"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n")
        lines = [line for line in lines if line]
        
        if not lines:
            # Also check for errors
            if result.returncode != 0 and result.stderr:
                return f"No files found matching '{pattern}' in {path}. Note: find exited with code {result.returncode}, stderr: {result.stderr[:200]}"
            return f"No files found matching '{pattern}' in {path}"
        
        # Limit results
        total_files = len(lines)
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({total_files - max_results} more results)")
        
        return f"Found {total_files} files matching '{pattern}':\n" + "\n".join(lines)
    
    except subprocess.TimeoutExpired:
        return f"Error: find search timed out after 30s. Try a more specific pattern or narrower directory."
    except Exception as e:
        return f"Error running find: {e}"


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = True,
) -> str:
    """Execute a search command."""
    try:
        search_path = _get_search_path(path)
        
        if command == "grep":
            return _run_grep(pattern, search_path, file_pattern, max_results, case_sensitive)
        elif command == "find":
            return _run_find(pattern, search_path, max_results)
        else:
            return f"Error: unknown command {command}"
    
    except Exception as e:
        return f"Error: {e}"
