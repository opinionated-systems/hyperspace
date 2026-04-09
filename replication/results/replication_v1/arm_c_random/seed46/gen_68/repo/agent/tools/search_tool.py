"""
Search tool: find files and search for patterns in the codebase.

Provides grep-like functionality and file finding capabilities
to help the meta agent locate code that needs modification.
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
            "Commands: grep (search for text pattern), find (find files by name). "
            "Useful for locating code that needs modification."
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
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
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
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
) -> str:
    """Execute a search command."""
    try:
        # Validate command
        valid_commands = ["grep", "find"]
        if command not in valid_commands:
            return f"Error: unknown command '{command}'. Valid commands: {valid_commands}"
        
        # Validate pattern
        if not pattern:
            return "Error: pattern is required"
        
        # Determine search root
        search_root = path or _ALLOWED_ROOT or os.getcwd()
        search_root = os.path.abspath(search_root)

        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_root.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search path {search_root} is outside allowed root {_ALLOWED_ROOT}"
        
        # Check if search root exists
        if not os.path.exists(search_root):
            return f"Error: Search path does not exist: {search_root}"
        if not os.path.isdir(search_root):
            return f"Error: Search path is not a directory: {search_root}"

        if command == "grep":
            return _grep(pattern, search_root, file_extension)
        elif command == "find":
            return _find(pattern, search_root, file_extension)
        else:
            return f"Error: unknown command {command}"
    except PermissionError as e:
        return f"Error: Permission denied - {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _grep(pattern: str, search_root: str, file_extension: str | None) -> str:
    """Search for pattern in files using grep."""
    # Build find command to get files
    find_cmd = ["find", search_root, "-type", "f"]
    
    if file_extension:
        find_cmd.extend(["-name", f"*{file_extension}"])
    
    # Exclude hidden directories and __pycache__
    find_cmd.extend(["-not", "-path", "*/\.*", "-not", "-path", "*/__pycache__/*"])
    
    try:
        # Get list of files
        result = subprocess.run(
            find_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        
        if not files or files == ['']:
            return f"No files found in {search_root}"
        
        # Run grep on files
        matches = []
        for f in files[:100]:  # Limit to first 100 files for performance
            if not f:
                continue
            try:
                grep_result = subprocess.run(
                    ["grep", "-n", "-H", "-i", pattern, f],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if grep_result.returncode == 0 and grep_result.stdout:
                    matches.append(grep_result.stdout)
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue
        
        if matches:
            output = "".join(matches)
            return _truncate(output, 8000)
        else:
            return f"No matches found for '{pattern}'"
            
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except Exception as e:
        return f"Error during search: {e}"


def _find(pattern: str, search_root: str, file_extension: str | None) -> str:
    """Find files by name pattern."""
    find_cmd = ["find", search_root, "-type", "f", "-name", f"*{pattern}*"]
    
    if file_extension:
        find_cmd[-1] = f"*{pattern}*{file_extension}"
    
    # Exclude hidden directories and __pycache__
    find_cmd.extend(["-not", "-path", "*/\.*", "-not", "-path", "*/__pycache__/*"])
    
    try:
        result = subprocess.run(
            find_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            files = result.stdout.strip().split("\n")
            return f"Found {len(files)} file(s):\n" + _truncate(result.stdout, 5000)
        else:
            return f"No files found matching '{pattern}'"
            
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except Exception as e:
        return f"Error during find: {e}"
