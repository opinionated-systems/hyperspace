"""
Search tool: find patterns in files using grep/ripgrep.

Provides file search capabilities to help agents locate code patterns,
function definitions, and specific text within the codebase.
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
            "Can search file contents or find files by name. "
            "Results are limited to prevent context overflow."
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
                    "description": "Directory or file to search in (absolute path). Default: allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50, max: 200).",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None
_MAX_RESULTS_DEFAULT = 50
_MAX_RESULTS_LIMIT = 200


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate_output(output: str, max_lines: int = 100) -> str:
    """Truncate output to prevent context overflow."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} more lines truncated] ..."
    return output


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    max_results: int | None = None,
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search (default: allowed root)
        file_pattern: Optional glob pattern to filter files (e.g., '*.py')
        max_results: Maximum results to return (default: 50, max: 200)

    Returns:
        Search results or error message
    """
    # Validate pattern
    if not pattern or not pattern.strip():
        return "Error: Empty search pattern provided"

    # Determine search path
    search_path = path or _ALLOWED_ROOT or os.getcwd()
    search_path = os.path.abspath(search_path)

    # Scope check
    if _ALLOWED_ROOT is not None:
        if not search_path.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

    # Validate path exists
    if not os.path.exists(search_path):
        return f"Error: Path does not exist: {search_path}"

    # Validate and cap max_results
    effective_max = _MAX_RESULTS_DEFAULT
    if max_results is not None:
        if max_results <= 0:
            return "Error: max_results must be positive"
        effective_max = min(max_results, _MAX_RESULTS_LIMIT)

    try:
        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*"]

        # Add pattern and path
        cmd.extend([pattern, search_path])

        # Run search with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in (0, 1):
            return f"Error: grep failed with code {result.returncode}: {result.stderr}"

        output = result.stdout

        if not output.strip():
            return f"No matches found for pattern '{pattern}' in {search_path}"

        # Count lines and truncate if needed
        lines = output.strip().split("\n")
        total_matches = len(lines)

        if total_matches > effective_max:
            output = "\n".join(lines[:effective_max])
            output += f"\n\n[Showing {effective_max} of {total_matches} matches. Use max_results parameter to see more.]"

        return _truncate_output(f"Found {total_matches} matches for '{pattern}':\n{output}")

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
    except FileNotFoundError:
        return "Error: grep command not found. Please ensure grep is installed."
    except Exception as e:
        return f"Error: Search failed: {type(e).__name__}: {e}"
