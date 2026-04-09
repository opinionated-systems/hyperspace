"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help the agent find code patterns,
function definitions, and references across the codebase.
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Dict, Any


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports searching for text patterns, file names, and code references. "
            "Results include file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to filter files (e.g., '*.py', '*.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: false).",
                },
            },
            "required": ["pattern"],
        },
    }


def _get_allowed_root() -> str | None:
    """Get the allowed root directory from bash_tool if available."""
    try:
        from agent.tools import bash_tool
        return getattr(bash_tool, '_ALLOWED_ROOT', None)
    except ImportError:
        return None


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in (default: current directory or allowed root)
        file_pattern: File glob pattern to filter files (e.g., '*.py')
        case_sensitive: Whether the search is case sensitive
    
    Returns:
        Formatted search results with file paths and line numbers
    """
    # Determine search root
    allowed_root = _get_allowed_root()
    if path is None:
        search_path = allowed_root or os.getcwd()
    else:
        search_path = os.path.abspath(path)
        # Ensure we stay within allowed root if set
        if allowed_root and not search_path.startswith(allowed_root):
            return f"Error: Search path '{search_path}' is outside allowed root '{allowed_root}'"
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist"
    
    # Build grep command
    cmd_parts = ["grep", "-r", "-n"]
    
    if not case_sensitive:
        cmd_parts.append("-i")
    
    # Add file pattern filter if specified
    if file_pattern:
        cmd_parts.extend(["--include", file_pattern])
    
    # Add pattern and path
    cmd_parts.extend([pattern, search_path])
    
    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                # Truncate if too many results
                truncated = lines[:50]
                output = "\n".join(truncated)
                output += f"\n\n... ({len(lines) - 50} more results truncated)"
            else:
                output = result.stdout.strip()
            return output if output else "No matches found"
        elif result.returncode == 1:
            # No matches found (grep returns 1 when no matches)
            return "No matches found"
        else:
            # Error occurred
            return f"Error: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
