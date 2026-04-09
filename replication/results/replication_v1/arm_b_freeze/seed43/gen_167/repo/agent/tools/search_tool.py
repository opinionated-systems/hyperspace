"""
Search tool: find content in files using grep-like functionality.

Provides file search capabilities to help locate code patterns,
function definitions, and specific text within the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns and can search recursively. "
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
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
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


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files."""
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
        cmd = ["grep", "-r", "-n", "-I"]
        
        # Add file extension filter if specified
        if file_extension:
            cmd.append(f"--include=*{file_extension}")
        
        # Add pattern
        cmd.append(pattern)
        cmd.append(search_path)

        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode not in (0, 1):  # 0 = matches found, 1 = no matches
            return f"Search error: {result.stderr}"

        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        if not lines or lines == ['']:
            return f"No matches found for pattern '{pattern}'"

        # Limit results
        total_found = len(lines)
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total_found - max_results} more results truncated)"
        else:
            truncated_msg = ""

        # Format results
        formatted = []
        for line in lines:
            if line:
                formatted.append(line)

        return f"Found {len(lines)} matches for '{pattern}':\n" + "\n".join(formatted) + truncated_msg

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern."
    except Exception as e:
        return f"Error during search: {e}"
