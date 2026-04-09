"""
Search tool: find patterns in files using grep/ripgrep.

Provides file search capabilities to help locate code patterns,
function definitions, and specific text within the codebase.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns, file type filtering, and case-insensitive search. "
            "Results are limited to avoid overwhelming output."
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
                    "description": "Directory or file to search in (absolute path). Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter by (e.g., '*.py', '*.js'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: true.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = True,
) -> str:
    """Search for pattern in files.
    
    Uses ripgrep if available, falls back to grep.
    """
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            if not Path(search_path).exists():
                return f"Error: Path does not exist: {path}"
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        # Build command - prefer ripgrep, fall back to grep
        cmd_parts = []
        
        # Check for ripgrep
        rg_available = subprocess.run(
            ["which", "rg"], capture_output=True, text=True
        ).returncode == 0
        
        if rg_available:
            cmd_parts.append("rg")
            if not case_sensitive:
                cmd_parts.append("-i")
            cmd_parts.append("--line-number")
            cmd_parts.append("--max-count=5")  # Limit matches per file
            cmd_parts.append("--max-filesize=1M")
            if file_pattern:
                cmd_parts.extend(["-g", file_pattern])
            cmd_parts.append(pattern)
            if Path(search_path).is_dir():
                cmd_parts.append(search_path)
            else:
                cmd_parts.append(search_path)
        else:
            # Fallback to grep
            cmd_parts.append("grep")
            cmd_parts.append("-r")
            if not case_sensitive:
                cmd_parts.append("-i")
            cmd_parts.append("-n")
            cmd_parts.append("--max-count=5")
            if file_pattern:
                cmd_parts.extend(["--include", file_pattern])
            cmd_parts.append(pattern)
            cmd_parts.append(search_path)
        
        # Run search
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 50:
                # Truncate if too many results
                truncated = lines[:50]
                return "Search results (truncated to 50 lines):\n" + "\n".join(truncated) + f"\n... ({len(lines) - 50} more lines)"
            return "Search results:\n" + result.stdout
        elif result.returncode == 1:
            return f"No matches found for pattern: {pattern}"
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"
