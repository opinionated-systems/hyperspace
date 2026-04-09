"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns and can search specific files or directories. "
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
                    "description": "Absolute path to file or directory to search. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: true.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = root


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = True,
) -> str:
    """Execute a search using grep."""
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: no path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = path
            # Scope check
            if _ALLOWED_ROOT is not None:
                resolved = Path(search_path).resolve()
                if not str(resolved).startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        p = Path(search_path)
        if not p.exists():
            return f"Error: {search_path} does not exist."

        # Build grep command
        cmd = ["grep", "-rn" if case_sensitive else "-rni"]
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
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
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                # Truncate if too many results
                truncated = lines[:50]
                return (
                    f"Found {len(lines)} matches (showing first 50):\n" +
                    "\n".join(truncated) +
                    f"\n... ({len(lines) - 50} more results)"
                )
            return f"Found {len(lines)} matches:\n" + result.stdout
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {search_path}"
        else:
            # Error
            return f"Error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30 seconds. Try a more specific pattern or narrower scope."
    except Exception as e:
        return f"Error: {e}"
