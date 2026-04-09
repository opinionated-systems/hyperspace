"""
Search tool: find patterns in files using grep-like functionality.

Provides content search capabilities to complement the editor tool.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": "Search for patterns in files using grep. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Default is current directory.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(pattern: str, path: str = ".", file_extension: str | None = None) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file path to search in
        file_extension: Optional extension filter (e.g., '.py')
    
    Returns:
        String with matching lines formatted as "file:line:content"
    """
    try:
        target = Path(path)
        
        # Build grep command
        cmd = ["grep", "-r", "-n", "-E", pattern]
        
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add target path
        cmd.append(str(target))
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Limit output to avoid overwhelming responses
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return result.stdout.strip()
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (30s limit)."
    except Exception as e:
        return f"Error during search: {e}"
