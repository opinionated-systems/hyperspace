"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep. Returns matching lines with file paths and line numbers. Supports regex patterns and can search specific file types.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js'). Use '*' for all files.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: True)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50, max: 200)",
                },
                "whole_word": {
                    "type": "boolean",
                    "description": "Match whole words only (default: False)",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 50,
    whole_word: bool = False,
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional file extension filter
        case_sensitive: Whether the search is case sensitive
        max_results: Maximum number of results to return
        whole_word: Match whole words only

    Returns:
        Matching lines with file paths and line numbers
    """
    # Validate and cap max_results
    max_results = min(max(1, max_results), 200)

    # Validate path exists
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]

        if not case_sensitive:
            cmd.append("-i")

        if whole_word:
            cmd.append("-w")

        # Add pattern (escape if needed for shell)
        cmd.append(pattern)

        # Add path
        cmd.append(path)

        # Add file extension filter if specified
        if file_extension and file_extension != "*":
            cmd.extend(["--include", f"*{file_extension}"])

        # Exclude binary files and common non-source directories
        cmd.extend(["--binary-files=without-match"])
        cmd.extend(["--exclude-dir", ".git"])
        cmd.extend(["--exclude-dir", "__pycache__"])
        cmd.extend(["--exclude-dir", "node_modules"])
        cmd.extend(["--exclude-dir", ".venv"])
        cmd.extend(["--exclude-dir", "venv"])

        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # grep returns exit code 1 when no matches found
        if result.returncode not in (0, 1):
            return f"Error: grep failed with exit code {result.returncode}: {result.stderr}"

        lines = result.stdout.strip().split("\n") if result.stdout else []

        if not lines or lines == [""]:
            return f"No matches found for pattern '{pattern}' in '{path}'"

        total_matches = len(lines)

        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({total_matches - max_results} more results, showing {max_results} of {total_matches})")

        header = f"Found {total_matches} match(es) for '{pattern}':\n"
        return header + "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
