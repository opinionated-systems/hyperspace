"""
Search tool: grep-like file search capabilities.

Provides text search within files to help navigate and understand codebases.
Supports regex patterns, file filtering, and context lines.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files using grep-like functionality. "
            "Supports regex patterns, file type filtering, and shows context lines. "
            "Useful for finding code patterns, function definitions, or references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter by (e.g., '*.py', '*.js'). Optional.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before/after matches. Default 2.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive. Default False.",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_allowed(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    context_lines: int = 2,
    case_sensitive: bool = False,
) -> str:
    """Execute a file search."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not _is_allowed(str(p)):
            return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {p} does not exist."

        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add context lines
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        
        # Case sensitivity
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(p))
        
        # Add file pattern if specified
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in (0, 1):
            return f"Error: search failed with code {result.returncode}: {result.stderr}"
        
        output = result.stdout
        
        if not output.strip():
            return f"No matches found for pattern '{pattern}' in {path}"
        
        # Truncate if too long
        lines = output.split("\n")
        if len(lines) > 100:
            output = "\n".join(lines[:50]) + f"\n... ({len(lines) - 100} lines omitted) ...\n" + "\n".join(lines[-50:])
        
        match_count = len([l for l in lines if l.strip() and not l.startswith("--")])
        return f"Found {match_count} matches for '{pattern}':\n{output}"
        
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30 seconds. Try a more specific pattern or smaller directory."
    except Exception as e:
        return f"Error: {e}"
