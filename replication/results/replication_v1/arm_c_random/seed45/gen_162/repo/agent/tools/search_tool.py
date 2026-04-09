"""
Search tool: search for patterns in files using grep/ripgrep.

Provides file search capabilities to help the agent explore codebases.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep. Searches file contents for matching patterns and returns matching lines with file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js'). If provided, only searches files with this extension.",
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
    file_extension: str = "",
    max_results: int = 50,
) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in
        file_extension: Optional file extension filter
        max_results: Maximum number of results to return
    
    Returns:
        Search results as formatted string
    """
    try:
        # Build the grep command
        cmd = ["grep", "-r", "-n", "-E", pattern]
        
        # Add file extension filter if provided
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add the search path
        cmd.append(path)
        
        # Run the search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Process results
        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        if not lines or lines == [""]:
            return f"No matches found for pattern '{pattern}'"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (truncated, showing {max_results} of {len(lines)} matches)"
        else:
            truncated_msg = ""
        
        # Format results
        formatted = []
        for line in lines:
            if line:
                formatted.append(line)
        
        header = f"Found {len(lines)} match(es) for pattern '{pattern}':\n"
        return header + "\n".join(formatted) + truncated_msg
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds"
    except FileNotFoundError:
        # Fallback if grep is not available
        return "Error: grep command not found"
    except Exception as e:
        return f"Error during search: {e}"
