"""
Search tool: search for patterns in files using grep.

Provides content-based file search capabilities.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


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
    """Search for patterns in files using grep.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file path to search in
        file_extension: Optional file extension filter
        case_sensitive: Whether the search should be case sensitive

    Returns:
        Matching lines with file paths and line numbers, or error message
    """
    try:
        search_path = Path(path).expanduser().resolve()
        
        if not search_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add case insensitive flag if needed
        if not case_sensitive:
            cmd.append("-i")
        
        # Add the pattern
        cmd.append(pattern)
        
        # Add file extension filter if provided
        if file_extension:
            if not file_extension.startswith("."):
                file_extension = "." + file_extension
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add the path
        cmd.append(str(search_path))
        
        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            lines = result.stdout.strip().split("\n")
            # Limit output to avoid overwhelming responses
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return result.stdout.strip()
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}'"
        else:
            # Error occurred
            return f"Error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (30s limit)"
    except Exception as e:
        return f"Error searching: {e}"
