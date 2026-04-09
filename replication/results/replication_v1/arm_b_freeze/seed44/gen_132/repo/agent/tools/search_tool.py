"""
Search tool: find patterns in files using grep and glob.

Provides fast file search capabilities to locate code patterns,
function definitions, and text across the codebase.
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
            "Fast way to find code patterns, function definitions, "
            "and text across the codebase. "
            "Returns matching lines with file paths and line numbers."
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
                    "description": "Directory or file to search in (absolute path). Default: allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to search (e.g., '*.py'). Default: all files.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: False.",
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


def _truncate_output(output: str, max_lines: int = 100, max_chars: int = 10000) -> str:
    """Truncate output if it exceeds limits."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"... [{len(lines)} total matches, truncated to {max_lines}]")
    result = "\n".join(lines)
    if len(result) > max_chars:
        half = max_chars // 2
        result = result[:half] + f"\n... [output truncated, {len(result)} chars total] ...\n" + result[-half:]
    return result


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.
    
    Uses ripgrep (rg) if available, falls back to grep.
    Returns matching lines with file:line:content format.
    """
    try:
        # Determine search path
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        search_path = os.path.abspath(search_path)
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        # Check if path exists
        if not os.path.exists(search_path):
            return f"Error: path does not exist: {search_path}"
        
        # Build command - prefer ripgrep for speed
        use_rg = _has_ripgrep()
        cmd = ["rg" if use_rg else "grep", "-n"]
        
        if not use_rg:
            cmd.append("-r")  # Recursive for grep
        
        if not case_sensitive:
            cmd.append("-i")  # Case insensitive
        
        if file_pattern:
            if use_rg:
                cmd.extend(["-g", file_pattern])
            else:
                # grep with include
                cmd.extend(["--include", file_pattern])
        
        # Add pattern and path
        cmd.extend([pattern, search_path])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            output = result.stdout
            if not output.strip():
                return f"No matches found for pattern: {pattern}"
            return _truncate_output(output)
        elif result.returncode == 1:
            # No matches (grep returns 1 for no matches)
            return f"No matches found for pattern: {pattern}"
        else:
            return f"Search error (exit code {result.returncode}): {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def _has_ripgrep() -> bool:
    """Check if ripgrep (rg) is available."""
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
