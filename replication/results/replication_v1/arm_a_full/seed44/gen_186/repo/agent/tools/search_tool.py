"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help the meta agent find code patterns,
function definitions, and references across the codebase.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns and can search specific files or directories. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default: false.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search (defaults to allowed root)
        file_extension: Optional extension filter (e.g., '.py')
        case_sensitive: Whether search is case sensitive
    
    Returns:
        Matching lines with file:line format
    """
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Exclude binary files and common non-source directories
        cmd.extend([
            "--binary-files=without-match",
            "--exclude-dir=.git",
            "--exclude-dir=__pycache__",
            "--exclude-dir=.pytest_cache",
            "--exclude-dir=node_modules",
            "--exclude-dir=venv",
            "--exclude-dir=.venv",
        ])
        
        cmd.extend([pattern, search_path])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            return _truncate(result.stdout.strip(), 8000)
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}'"
        else:
            # Error
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (pattern may be too broad)"
    except Exception as e:
        return f"Error: {e}"
