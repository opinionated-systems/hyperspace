"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help the agent explore and find
code patterns, function definitions, and references across the codebase.
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
            "Supports regex patterns, file filtering, and recursive search. "
            "Useful for finding code patterns, function definitions, and references."
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
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern filter (e.g., '*.py', '*.js').",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories.",
                    "default": True,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive.",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 50,
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


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search for patterns in files."""
    try:
        # Validate pattern is not empty
        if not pattern or not pattern.strip():
            return "Error: pattern cannot be empty"
        
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {p} does not exist."

        # Build grep command
        cmd = ["grep"]
        
        # Add options
        if not case_sensitive:
            cmd.append("-i")  # Case insensitive
        if recursive and p.is_dir():
            cmd.append("-r")
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Add line numbers and limit results
        cmd.extend(["-n", "-m", str(max_results)])
        
        # Add pattern and path
        cmd.extend([pattern, str(p)])

        # Execute search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Found matches
            output = result.stdout
            if len(output) > 10000:
                output = output[:5000] + "\n... [output truncated] ...\n" + output[-5000:]
            lines = output.count("\n")
            return f"Found {lines} matches:\n{output}"
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {p}"
        else:
            # Error
            return f"Search error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or smaller directory."
    except Exception as e:
        return f"Error: {e}"
