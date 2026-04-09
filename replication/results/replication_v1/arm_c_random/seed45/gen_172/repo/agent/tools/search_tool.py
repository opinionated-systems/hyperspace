"""Search tool for finding patterns in files using grep."""

from __future__ import annotations

import subprocess
from typing import Any


def search_in_files(pattern: str, path: str = ".", file_extension: str | None = None) -> str:
    """Search for a pattern in files using grep.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional file extension filter (e.g., '.py')

    Returns:
        Search results with matching lines and line numbers
    """
    try:
        cmd = ["grep", "-rn", "--include", f"*{file_extension or ''}", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return result.stdout
        elif result.returncode == 1:
            return "No matches found"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def search_file_content(file_path: str, pattern: str) -> str:
    """Search for a pattern within a specific file.

    Args:
        file_path: Path to the file to search
        pattern: The regex pattern to search for

    Returns:
        Matching lines with line numbers
    """
    try:
        cmd = ["grep", "-n", pattern, file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout
        elif result.returncode == 1:
            return "Pattern not found in file"
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error: {e}"


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_in_files",
        "description": "Search for a pattern in files using grep. Returns matching lines with line numbers. Limited to 50 results.",
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
                    "description": "Optional file extension filter (e.g., '.py')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search_file_content",
        "description": "Search for a pattern within a specific file. Returns matching lines with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to search",
                },
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
            },
            "required": ["file_path", "pattern"],
        },
    },
]


def get_tools() -> list[dict[str, Any]]:
    """Return tool definitions with their functions."""
    return [
        {
            "info": TOOL_DEFINITIONS[0],
            "function": search_in_files,
        },
        {
            "info": TOOL_DEFINITIONS[1],
            "function": search_file_content,
        },
    ]
