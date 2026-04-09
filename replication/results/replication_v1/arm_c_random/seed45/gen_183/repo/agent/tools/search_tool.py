"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns and can search recursively. "
            "Returns matching lines with file names and line numbers."
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
                    "description": "Directory or file to search in. Defaults to current directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories. Default: true.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Filter by file extension (e.g., '.py', '.js'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search. Default: false (case-insensitive).",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    recursive: bool = True,
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Execute a search command using grep."""
    try:
        # Determine search path
        if path is None:
            search_path = _ALLOWED_ROOT or os.getcwd()
        else:
            search_path = os.path.abspath(path)

        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        # Build grep command
        cmd = ["grep"]
        
        # Add options
        if recursive:
            cmd.append("-r")
        if not case_sensitive:
            cmd.append("-i")
        cmd.append("-n")  # Always show line numbers
        cmd.append("--color=never")  # Disable color codes
        
        # Add pattern
        cmd.append(pattern)
        
        # Add search path
        cmd.append(search_path)
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Execute search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Process results
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 100:
                # Truncate if too many results
                truncated = "\n".join(lines[:50])
                truncated += f"\n... ({len(lines) - 100} more results) ...\n"
                truncated += "\n".join(lines[-50:])
                return f"Found {len(lines)} matches:\n{truncated}"
            return f"Found {len(lines)} matches:\n{result.stdout}"
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}'"
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern."
    except Exception as e:
        return f"Error: {e}"
