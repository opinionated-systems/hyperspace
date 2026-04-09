"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns
within files in the repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    """Return search tool metadata."""
    return {
        "name": "search",
        "description": "Search for a pattern in files within a directory. Uses grep-like syntax. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt')",
                },
            },
            "required": ["pattern", "path"],
        },
    }


def tool_function(pattern: str, path: str, file_extension: str | None = None) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file path to search in
        file_extension: Optional file extension filter
        
    Returns:
        Matching lines with file paths and line numbers
    """
    target_path = Path(path)
    
    if not target_path.exists():
        return f"Error: Path '{path}' does not exist"
    
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n", "-E", pattern]
        
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add target path
        cmd.append(str(target_path))
        
        # Run search
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
        return "Error: Search timed out (30s limit)"
    except Exception as e:
        return f"Error during search: {e}"
