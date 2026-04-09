"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to the agent.
"""

from __future__ import annotations

import subprocess
import logging

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool info for the search tool."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep. Returns matching lines with file paths and line numbers. Useful for finding code patterns, function definitions, or specific text across the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in. Defaults to current directory.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js'). If provided, only searches files with this extension.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(pattern: str, path: str = ".", file_extension: str | None = None) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory or file path to search in
        file_extension: Optional file extension filter
        
    Returns:
        Matching lines with file paths and line numbers
    """
    try:
        # Build grep command
        cmd = ["grep", "-rn", "--include=*"]
        
        if file_extension:
            cmd = ["grep", "-rn", f"--include=*{file_extension}"]
        
        cmd.extend([pattern, path])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                # Truncate if too many results
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return result.stdout
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}'"
        else:
            # Error
            return f"Error searching: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
