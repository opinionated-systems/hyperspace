"""
Search tool: search for patterns in files using grep/ripgrep.

Provides text search capabilities across the codebase.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for text patterns in files using grep. Returns matching lines with file paths and line numbers. Useful for finding code patterns, function definitions, or references across the codebase.",
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
                    "description": "Whether the search is case sensitive (default: False)",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional file extension filter
        case_sensitive: Whether the search is case sensitive

    Returns:
        Matching lines with file paths and line numbers
    """
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(path)
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                # Truncate if too many results
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return result.stdout.strip()
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds."
    except FileNotFoundError:
        # Fallback if grep is not available
        return "Error: grep command not found."
    except Exception as e:
        return f"Error during search: {e}"
