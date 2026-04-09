"""
Search tool: search for patterns in files using grep/find.

Provides file search capabilities for the meta agent to locate
code patterns, function definitions, and specific text.
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
            "Can search for text patterns, function definitions, or specific code. "
            "Returns matching file paths with line numbers and context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex or plain text).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py'). Defaults to all files.",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Whether pattern is a regex. Default: false (plain text).",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case sensitive search. Default: false.",
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
    file_pattern: str | None = None,
    is_regex: bool = False,
    case_sensitive: bool = False,
) -> str:
    """Search for pattern in files."""
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No search path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            # Scope check
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add context lines
        cmd.extend(["-B", "2", "-A", "2"])
        
        # Case sensitivity
        if not case_sensitive:
            cmd.append("-i")
        
        # Regex or fixed string
        if is_regex:
            cmd.append("-E")
        else:
            cmd.append("-F")
        
        # File pattern
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Exclude binary files and hidden directories
        cmd.extend(["-I", "--exclude-dir=.*"])
        
        # Pattern and path
        cmd.extend([pattern, search_path])

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
                truncated = "\n".join(lines[:50])
                return f"Found {len(lines)} matches (showing first 50):\n{truncated}\n... (truncated)"
            return f"Found matches:\n{result.stdout}"
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for '{pattern}' in {search_path}"
        else:
            # Error
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern."
    except Exception as e:
        return f"Error: {e}"
