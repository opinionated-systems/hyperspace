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
        # Validate pattern
        if not pattern or not pattern.strip():
            return "Error: Pattern cannot be empty"
        
        pattern = pattern.strip()

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

        # Check if search path exists
        if not os.path.exists(search_path):
            return f"Error: Path does not exist: {search_path}"

        # Build grep command
        cmd = ["grep", "-r", "-n", "-I"]
        
        # Add file extension filter if provided
        if file_extension:
            # Clean up extension (remove leading dot if present)
            ext = file_extension.lstrip('.')
            cmd.append(f"--include=*.{ext}")
        
        # Add pattern - use fixed strings for simple patterns to avoid regex issues
        if pattern.isalnum() or all(c.isalnum() or c in '_- ' for c in pattern):
            cmd.insert(1, "-F")  # Fixed string matching
        
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
            stderr = result.stderr.strip()
            if "No such file or directory" in stderr:
                return f"Error: Search path not found: {search_path}"
            return f"Search error: {stderr}"

        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        # Filter out empty lines
        lines = [line for line in lines if line.strip()]
        
        if not lines:
            return f"No matches found for pattern '{pattern}'"

        total_matches = len(lines)
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total_matches - max_results} more results truncated)"
        else:
            truncated_msg = ""

        # Format results with better readability
        formatted = []
        for line in lines:
            # Highlight the match in the line
            formatted.append(line)

        return f"Found {total_matches} matches for '{pattern}':\n" + "\n".join(formatted) + truncated_msg

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or narrower file filter."
    except re.error as e:
        return f"Error: Invalid regex pattern '{pattern}': {e}. Try using a simpler pattern."
    except Exception as e:
        return f"Error during search: {type(e).__name__}: {e}"
