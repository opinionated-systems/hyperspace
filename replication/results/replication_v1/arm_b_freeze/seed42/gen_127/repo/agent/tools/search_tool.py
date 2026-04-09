"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep. Returns matching lines with file paths and line numbers. Supports context lines for better code understanding.",
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
                    "description": "Optional file extension filter (e.g., '.py', '.js')",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: True)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before and after each match (default: 0)",
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
    context_lines: int = 0,
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional file extension filter
        case_sensitive: Whether the search is case sensitive
        max_results: Maximum number of results to return
        context_lines: Number of context lines to show before and after each match

    Returns:
        Matching lines with file paths and line numbers
    """
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]

        if not case_sensitive:
            cmd.append("-i")

        # Add context lines if specified
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        # Add pattern
        cmd.append(pattern)

        # Add path
        cmd.append(path)

        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])

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
            return f"No matches found for pattern '{pattern}'"

        # Limit results (count match lines, not context lines)
        if context_lines == 0:
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more results)")
        else:
            # With context lines, we need to be more careful about limiting
            # Count actual matches (lines without leading "-" or empty lines between groups)
            match_count = 0
            limited_lines = []
            for line in lines:
                if line and not line.startswith("-"):
                    match_count += 1
                if match_count > max_results and line and not line.startswith("-"):
                    limited_lines.append(f"\n... (more results)")
                    break
                limited_lines.append(line)
            lines = limited_lines

        return "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
