"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
"""

from __future__ import annotations

import os
import subprocess


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Returns matching lines with file paths and line numbers. "
            "Useful for finding code patterns, function definitions, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The regex pattern to search for.
        path: Directory or file to search in (default: current directory).
        file_extension: Optional file extension filter (e.g., '.py').

    Returns:
        Matching lines with file paths and line numbers.
    """
    search_path = path or "."
    
    # Build grep command
    cmd = ["grep", "-rn", "--include", file_extension or "*", pattern, search_path]
    
    if file_extension:
        cmd = ["grep", "-rn", "--include", f"*{file_extension}", pattern, search_path]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            # Limit output length
            if len(output) > 5000:
                output = output[:5000] + "\n... (output truncated)"
            return output if output else "No matches found."
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"
