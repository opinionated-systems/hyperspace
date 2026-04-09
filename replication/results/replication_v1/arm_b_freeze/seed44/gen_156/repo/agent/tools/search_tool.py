"""
Search tool: search for patterns in files.

Provides grep-like functionality to find text patterns across files.
Useful for exploring codebases and finding relevant code sections.
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
            "Supports regex patterns and can search recursively. "
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
                    "description": "Directory or file to search in. Defaults to current directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories. Default: true.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search. Default: false (case-insensitive).",
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
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files."""
    try:
        # Default to current directory if no path provided
        search_path = path or "."
        p = Path(search_path)

        # Ensure absolute path
        if not p.is_absolute():
            p = Path(os.getcwd()) / p

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        # Build grep command
        cmd = ["grep"]
        
        # Add options
        if not case_sensitive:
            cmd.append("-i")  # Case insensitive
        if recursive:
            cmd.append("-r")
        cmd.append("-n")  # Line numbers
        cmd.append("-H")  # Print filename
        cmd.append("--include=*.py")  # Only Python files by default
        
        # Add pattern and path
        cmd.append(pattern)
        cmd.append(str(p))

        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Process results
        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        # Filter empty lines and limit results
        lines = [line for line in lines if line.strip()]
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({len(lines)} total matches, showing first {max_results}) ...")

        if not lines:
            return f"No matches found for pattern: {pattern}"

        return f"Search results for '{pattern}':\n" + "\n".join(lines)

    except subprocess.TimeoutExpired:
        return f"Error: Search timed out. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error during search: {e}"
