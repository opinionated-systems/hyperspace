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
        "description": "Search for patterns in files using grep. Returns matching lines with file paths and line numbers. Supports regex patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in (default: current directory)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., 'py', 'js', 'txt')",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: true)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
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
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory or file path to search in
        file_extension: Optional file extension filter
        case_sensitive: Whether the search is case sensitive
        max_results: Maximum number of results to return
    
    Returns:
        String with matching lines formatted as "file:line:content"
    """
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add case insensitive flag if needed
        if not case_sensitive:
            cmd.append("-i")
        
        # Add the pattern
        cmd.append(pattern)
        
        # Add file extension filter if specified
        if file_extension:
            cmd.append("--include")
            cmd.append(f"*.{file_extension}")
        
        # Add the path
        cmd.append(path)
        
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
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... and {len(result.stdout.strip().split(chr(10))) - max_results} more matches")
        
        return "\n".join(lines)
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error during search: {e}"
