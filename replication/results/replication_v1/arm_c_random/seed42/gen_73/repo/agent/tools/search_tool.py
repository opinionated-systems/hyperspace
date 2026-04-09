"""
Search tool: search for patterns in files using grep.

Provides file content search capabilities to complement the editor tool.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns and can search recursively."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for.",
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
                    "description": "Whether the search is case sensitive (default: True).",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = Path(root).resolve()


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    case_sensitive: bool = True,
) -> str:
    """Execute a search using grep."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = Path(path).resolve()
            try:
                resolved.relative_to(_ALLOWED_ROOT)
            except ValueError:
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        # Build grep command
        cmd = ["grep", "-r" if p.is_dir() else "", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(p))
        
        # Filter by extension if specified
        if file_extension and p.is_dir():
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Remove empty strings from cmd
        cmd = [c for c in cmd if c]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                return (
                    f"Found {len(lines)} matches (showing first 50):\n" +
                    "\n".join(lines[:50]) +
                    f"\n... and {len(lines) - 50} more matches"
                )
            return f"Found {len(lines)} matches:\n" + result.stdout
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            return f"Error: {result.stderr}"
            
    except Exception as e:
        return f"Error: {e}"
