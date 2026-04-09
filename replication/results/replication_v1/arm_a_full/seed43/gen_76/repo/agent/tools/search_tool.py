"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to find code patterns, function definitions,
and references across the codebase.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns, file type filtering, and recursive search. "
            "Useful for finding function definitions, imports, or specific code patterns."
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
                    "description": "Directory or file to search in (absolute path). Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter by (e.g., '*.py', '*.js'). Optional.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively. Default: true.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
            },
            "required": ["pattern"],
        },
    }


def _truncate_output(output: str, max_lines: int = 50) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} more lines truncated]"
    return output


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search (defaults to allowed root)
        file_pattern: Optional file pattern filter (e.g., '*.py')
        recursive: Whether to search recursively
        case_sensitive: Whether search is case sensitive
        max_results: Maximum number of results to return

    Returns:
        Search results with file paths and matching lines
    """
    # Validate pattern
    if not pattern or not pattern.strip():
        return "Error: Empty search pattern provided."

    # Determine search path
    search_path = path
    if search_path is None:
        if _ALLOWED_ROOT is None:
            return "Error: No search path provided and no allowed root set."
        search_path = _ALLOWED_ROOT

    p = Path(search_path)
    if not p.is_absolute():
        return f"Error: {search_path} is not an absolute path."

    # Scope check
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

    # Build grep command
    cmd = ["grep"]

    # Add options
    if recursive:
        cmd.append("-r")
    if not case_sensitive:
        cmd.append("-i")
    cmd.append("-n")  # Line numbers
    cmd.append("-H")  # Print filename

    # Add pattern
    cmd.append(pattern)

    # Add path
    cmd.append(str(p))

    # Add file pattern if specified
    if file_pattern:
        cmd.extend(["--include", file_pattern])

    # Exclude hidden directories
    cmd.extend(["--exclude-dir", ".*"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # grep returns exit code 1 when no matches found
        if result.returncode not in (0, 1):
            return f"Error: grep failed with exit code {result.returncode}: {result.stderr}"

        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern '{pattern}' in {search_path}"

        # Truncate if too many results
        truncated = _truncate_output(output, max_results)
        total_lines = len(output.split("\n"))

        if total_lines > max_results:
            return f"Found {total_lines} matches for '{pattern}':\n{truncated}"
        return f"Found {total_lines} matches for '{pattern}':\n{truncated}"

    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error during search: {e}"
