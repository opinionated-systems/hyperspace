"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents find code patterns,
function definitions, and text occurrences across the codebase.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata matching the paper's schema."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep. Supports regex patterns, file filtering, and case-insensitive search. Returns matching lines with line numbers.",
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
                    "default": ".",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py', '*.js')",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default: True)",
                    "default": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                    "default": 50,
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files using grep.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        file_pattern: Optional glob pattern to filter files
        case_sensitive: Whether search is case-sensitive
        max_results: Maximum number of results to return
    
    Returns:
        String with matching lines and line numbers, or error message
    """
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add case-insensitive flag if needed
        if not case_sensitive:
            cmd.append("-i")
        
        # Add the pattern
        cmd.append(pattern)
        
        # Add the path
        cmd.append(path)
        
        # Add file pattern if specified
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Run the search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Process results
        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        # Filter out empty lines
        lines = [line for line in lines if line.strip()]
        
        if not lines:
            return f"No matches found for pattern: {pattern}"
        
        # Limit results
        total_matches = len(lines)
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total_matches - max_results} more matches truncated)"
        else:
            truncated_msg = ""
        
        # Format output
        output = f"Found {total_matches} match(es) for pattern: {pattern}\n"
        output += "-" * 50 + "\n"
        for line in lines:
            output += line + "\n"
        output += truncated_msg
        
        return output
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error searching for pattern: {e}"
