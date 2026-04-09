"""
Search tool: search for patterns in files using grep/ripgrep.

Provides file search capabilities to help agents find code patterns,
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
            "Searches file contents for matching patterns. "
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
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None
_MAX_RESULTS_DEFAULT = 50


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_allowed_root(path: str) -> bool:
    """Check if path is within the allowed root directory."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int | None = None,
) -> str:
    """Search for pattern in files.
    
    Uses grep for pattern matching with optional file extension filtering.
    Returns matching lines with file paths and line numbers.
    """
    max_results = max_results or _MAX_RESULTS_DEFAULT
    
    # Determine search path
    search_path = path or _ALLOWED_ROOT or "."
    
    # Security check
    if not _is_within_allowed_root(search_path):
        return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    # Validate pattern (basic safety check)
    if not pattern or len(pattern) > 1000:
        return "Error: invalid pattern (empty or too long)"
    
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include", file_extension or "*"]
        
        # Add pattern and path
        cmd.extend([pattern, search_path])
        
        # Run search with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Process results
        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        if not lines or lines == ['']:
            return f"No matches found for pattern '{pattern}'"
        
        # Limit results
        total_matches = len(lines)
        if total_matches > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total_matches - max_results} more matches truncated)"
        else:
            truncated_msg = ""
        
        # Format output
        output = f"Found {total_matches} match(es) for pattern '{pattern}':\n"
        output += "\n".join(lines)
        output += truncated_msg
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: search timed out after 30 seconds"
    except Exception as e:
        return f"Error during search: {type(e).__name__}: {e}"
