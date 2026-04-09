"""
Search tool: search for patterns in files using grep.

Provides content-based search capabilities to complement the file tool.
"""

from __future__ import annotations

import subprocess
import logging

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool metadata."""
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
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive. Defaults to True.",
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
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file path to search in
        file_extension: Optional file extension filter
        case_sensitive: Whether search is case sensitive

    Returns:
        Matching lines with file paths and line numbers
    """
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add case insensitive flag if needed
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(path)
        
        # Add file extension filter if provided
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns exit code 1 when no matches found, which is not an error
        if result.returncode == 0:
            output = result.stdout
        elif result.returncode == 1:
            output = "No matches found."
        else:
            output = f"Error: {result.stderr}"
        
        # Limit output length
        if len(output) > 10000:
            output = output[:10000] + "\n... (output truncated)"
        
        return output
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"
