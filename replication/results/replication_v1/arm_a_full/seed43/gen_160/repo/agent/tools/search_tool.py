"""
Search tool: find patterns in files using grep/ripgrep.

Provides fast code search capabilities for the meta agent
to locate files and code patterns when modifying the codebase.
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
            "Fast way to find code across the codebase. "
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
                    "description": "Directory or file to search in (absolute path). Default: allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
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
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.
    
    Uses ripgrep if available, falls back to grep.
    Returns formatted results with file paths and line numbers.
    """
    try:
        # Determine search path
        search_path = path or _ALLOWED_ROOT or "."
        search_path = os.path.abspath(search_path)
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        # Check if path exists
        p = Path(search_path)
        if not p.exists():
            return f"Error: path does not exist: {search_path}"
        
        # Build command - prefer ripgrep, fall back to grep
        cmd = []
        
        # Try ripgrep first
        rg_available = subprocess.run(
            ["which", "rg"], capture_output=True
        ).returncode == 0
        
        if rg_available:
            cmd = ["rg", "--line-number", "--no-heading", "--with-filename"]
            if file_extension:
                cmd.extend(["--type-add", f"custom:*.{file_extension}", "-tcustom"])
            cmd.extend(["--max-count", str(max_results // 5 + 1)])  # Per file limit
            cmd.extend([pattern, search_path])
        else:
            # Fall back to grep
            cmd = ["grep", "-r", "-n", "--include", f"*{file_extension or ''}"]
            if file_extension:
                cmd[-1] = f"*.{file_extension}"
            cmd.extend([pattern, search_path])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Process results
        lines = result.stdout.strip().split("\n") if result.stdout else []
        lines = [l for l in lines if l.strip()]
        
        if not lines:
            return f"No matches found for '{pattern}' in {search_path}"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (truncated, showing {max_results} of {len(lines)}+ matches)"
        else:
            truncated_msg = ""
        
        # Format output
        output = f"Search results for '{pattern}' in {search_path}:\n"
        output += "\n".join(lines)
        output += truncated_msg
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: search timed out after 30s for pattern '{pattern}'"
    except Exception as e:
        return f"Error during search: {e}"
