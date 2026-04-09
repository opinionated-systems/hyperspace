"""
Search tool: find text patterns in files.

Provides grep-like functionality to search for patterns across the codebase.
Useful for finding code locations before editing.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from agent.config import DEFAULT_AGENT_CONFIG


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files. "
            "Uses grep-like syntax for finding code. "
            "Returns matching lines with file paths and line numbers."
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
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive (default: true).",
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


def _truncate_results(results: str, max_lines: int = 100) -> str:
    """Truncate results to max_lines."""
    lines = results.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return results


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    case_sensitive: bool = True,
) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional extension filter (e.g., '.py')
        case_sensitive: Whether search is case sensitive
    
    Returns:
        Matching lines with file:line format
    """
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        # Build grep command
        cmd = ["grep", "-rn" if case_sensitive else "-rni"]
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add pattern and path
        cmd.extend([pattern, str(p)])

        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Found matches
            return _truncate_results(result.stdout.strip())
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            # Error
            return f"Error searching: {result.stderr}"

    except Exception as e:
        return f"Error: {e}"
