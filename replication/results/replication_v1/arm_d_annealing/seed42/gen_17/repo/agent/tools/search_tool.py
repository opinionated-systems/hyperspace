"""
Search tool: search for patterns in files using grep.

Provides file content search capabilities to help agents explore codebases.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports searching within specific directories or files. "
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
                    "description": "Absolute path to directory or file to search in. Defaults to current directory.",
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
    _ALLOWED_ROOT = Path(root).resolve()


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = True,
) -> str:
    """Execute a search using grep."""
    try:
        # Default to current directory if no path provided
        if path is None:
            path = "."
        
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = p.resolve()
            if not str(resolved).startswith(str(_ALLOWED_ROOT)):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(p))
        
        # Add file extension filter if provided
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
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
            if len(lines) > 100:
                # Truncate if too many results
                truncated = lines[:50] + [f"... ({len(lines) - 100} more matches) ..."] + lines[-50:]
                return "Search results (truncated):\n" + "\n".join(truncated)
            return f"Search results ({len(lines)} matches):\n" + result.stdout
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            # Error occurred
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: search timed out (max 30 seconds)"
    except Exception as e:
        return f"Error: {e}"
