"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore and understand codebases.
Supports regex pattern matching, file filtering, and result limiting.
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
            "Supports regex patterns, file type filtering, and result limiting. "
            "Useful for finding code patterns, function definitions, or references."
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
                    "description": "Optional glob pattern to filter files (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
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
    max_results: int = 50,
) -> str:
    """Execute a search for patterns in files."""
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
            return f"Error: {p} does not exist."

        # Build grep command
        cmd = ["grep", "-r", "-n", "-E", pattern]
        
        # Add file pattern filter if provided
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Add path to search
        cmd.append(str(p))

        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # grep returns exit code 1 when no matches found, which is not an error
        if result.returncode not in [0, 1]:
            return f"Error: grep failed with exit code {result.returncode}: {result.stderr}"

        output = result.stdout
        if not output:
            return f"No matches found for pattern '{pattern}' in {path}"

        # Limit results
        lines = output.strip().split("\n")
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more results truncated)")

        formatted_output = "\n".join(lines)
        return _truncate(f"Found {len(lines)} matches for pattern '{pattern}':\n{formatted_output}")

    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error: {e}"
