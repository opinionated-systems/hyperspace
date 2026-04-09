"""
Search tool: search for patterns in files using grep.

Provides file search capabilities for the meta agent to find code patterns,
function definitions, and references across the codebase.
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
            "Useful for finding function definitions, imports, or specific code patterns. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex or literal string).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', '*.js'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive. Default: false.",
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


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Add pattern and path
        cmd.extend([pattern, str(p)])

        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Found matches
            return _truncate(result.stdout, 5000)
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            # Error
            return f"Error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or smaller directory."
    except Exception as e:
        return f"Error: {e}"
