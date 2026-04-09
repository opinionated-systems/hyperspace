"""
Search tool: search for patterns in files using grep/ripgrep.

Provides file search capabilities to help locate code for modification.
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
            "Supports regex patterns and can search within specific directories. "
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
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
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


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Execute a search for pattern in files."""
    try:
        # Determine search path
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        p = Path(search_path)
        
        if not p.is_absolute():
            return f"Error: {search_path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(p))
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Exclude hidden directories and __pycache__
        cmd.extend(["--exclude-dir", ".*", "--exclude-dir", "__pycache__"])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            output = _truncate(result.stdout, 8000)
            return f"Search results for '{pattern}':\n{output}"
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for '{pattern}' in {search_path}"
        else:
            # Error
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error: {e}"
