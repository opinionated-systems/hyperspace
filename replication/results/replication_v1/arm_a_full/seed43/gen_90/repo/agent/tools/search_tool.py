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
            "Returns matching lines with line numbers."
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
                    "description": "File or directory to search in. Defaults to current directory.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default is True.",
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
    path: str = ".",
    file_extension: str | None = None,
    case_sensitive: bool = True,
) -> str:
    """Search for a pattern in files."""
    try:
        p = Path(path)
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = str(p.resolve())
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        # Build grep command
        cmd = ["grep", "-n", "-r"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add pattern and path
        cmd.extend([pattern, str(p)])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                # Truncate if too many results
                return (
                    f"Found {len(lines)} matches (showing first 25 and last 25):\n\n"
                    + "\n".join(lines[:25])
                    + "\n...\n"
                    + "\n".join(lines[-25:])
                )
            return f"Found {len(lines)} matches:\n\n" + result.stdout
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (30s limit). Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error: {e}"
