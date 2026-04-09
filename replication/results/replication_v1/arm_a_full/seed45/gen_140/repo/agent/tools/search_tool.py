"""
Search tool: search for patterns in files using grep/ripgrep.

Provides file search capabilities to help the meta agent find code patterns,
function definitions, and references when modifying the codebase.

Features:
- Search for text patterns in files
- Search within specific file types
- Case-sensitive and case-insensitive search
- Line-numbered results with context
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
            "Returns line-numbered results with file paths. "
            "Useful for finding code patterns, function definitions, and references."
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
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive. Default: false",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50",
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
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in (defaults to allowed root)
        file_extension: Optional file extension filter (e.g., '.py')
        case_sensitive: Whether search is case-sensitive
        max_results: Maximum number of results to return
    
    Returns:
        Search results with file paths and line numbers
    """
    if not pattern:
        return "Error: pattern cannot be empty"
    
    # Determine search path
    search_path = path or _ALLOWED_ROOT or "."
    search_path = os.path.abspath(search_path)
    
    # Scope check
    if _ALLOWED_ROOT is not None:
        if not search_path.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    # Build grep command
    cmd = ["grep", "-r", "-n"]
    
    # Add case sensitivity flag
    if not case_sensitive:
        cmd.append("-i")
    
    # Add max results limit
    cmd.extend(["-m", str(max_results)])
    
    # Add pattern
    cmd.append(pattern)
    
    # Add file extension filter if specified
    if file_extension:
        if not file_extension.startswith("."):
            file_extension = "." + file_extension
        cmd.extend(["--include", f"*{file_extension}"])
    
    # Add search path
    cmd.append(search_path)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns exit code 1 when no matches found
        if result.returncode not in [0, 1]:
            return f"Error: search failed with exit code {result.returncode}: {result.stderr}"
        
        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern '{pattern}'"
        
        # Format results
        lines = output.split("\n")
        if len(lines) > max_results:
            lines = lines[:max_results]
            output = "\n".join(lines) + f"\n... ({len(lines)} results shown, more may exist)"
        
        return f"Found {len(lines)} match(es) for pattern '{pattern}':\n{output}"
    
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
