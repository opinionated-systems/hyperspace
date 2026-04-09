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
        # Determine search root
        search_root = path or _ALLOWED_ROOT or os.getcwd()
        search_root = os.path.abspath(search_root)

        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_root.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if command == "grep":
            return _grep(pattern, search_root, file_extension)
        elif command == "find":
            return _find(pattern, search_root, file_extension)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, search_root: str, file_extension: str | None) -> str:
    """Search for pattern in files using grep with xargs for efficiency."""
    # Build find command to get files
    find_cmd = ["find", search_root, "-type", "f"]
    
    if file_extension:
        find_cmd.extend(["-name", f"*{file_extension}"])
    
    # Exclude hidden directories and __pycache__
    find_cmd.extend(["-not", "-path", "*/\.*", "-not", "-path", "*/__pycache__/*"])
    
    try:
        # Use find | head -n 100 | xargs grep for efficient batch processing
        # This is much faster than running grep on each file individually
        find_proc = subprocess.Popen(
            find_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        
        # Limit to first 100 files for performance
        head_proc = subprocess.Popen(
            ["head", "-n", "100"],
            stdin=find_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        find_proc.stdout.close()  # Allow find_proc to receive a SIGPIPE if head_proc exits
        
        # Run grep on all files at once using xargs
        grep_result = subprocess.run(
            ["xargs", "grep", "-n", "-H", "-i", "-s", pattern],
            stdin=head_proc.stdout,
            capture_output=True,
            text=True,
            timeout=30,
        )
        head_proc.stdout.close()
        
        # Wait for find_proc to avoid zombie processes
        find_proc.wait()
        
        if grep_result.returncode == 0 and grep_result.stdout:
            return _truncate(grep_result.stdout, 8000)
        elif grep_result.returncode == 123:  # xargs exit code when some files can't be opened
            # Some files couldn't be read, but we may still have results
            if grep_result.stdout:
                return _truncate(grep_result.stdout, 8000)
            return f"No matches found for '{pattern}'"
        elif grep_result.returncode == 1:  # grep exit code when no matches found
            return f"No matches found for '{pattern}'"
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
