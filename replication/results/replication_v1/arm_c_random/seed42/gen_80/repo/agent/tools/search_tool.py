"""
Search tool: search for patterns in files using grep.

Provides file content search capabilities to help agents find code patterns.
"""

from __future__ import annotations

import subprocess
import os


def tool_info() -> dict:
    return {
        "name": "search",
        "description": "Search for patterns in files using grep. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The pattern to search for (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js')",
                },
            },
            "required": ["pattern", "path"],
        },
    }


def tool_function(pattern: str, path: str, file_extension: str = "") -> str:
    """Search for pattern in files.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional file extension filter
    
    Returns:
        String with matching lines (file:line:content format)
    """
    try:
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist"
        
        # Build grep command
        cmd = ["grep", "-rn", "--include", f"*{file_extension}" if file_extension else "*", pattern, path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            lines = result.stdout.strip().split("\n")
            # Limit output to avoid overwhelming the agent
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return result.stdout
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}'"
        else:
            # Error
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"
