"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents navigate codebases.
"""

from __future__ import annotations

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
                    "description": "Directory or file to search in (absolute path).",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories.",
                    "default": True,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search.",
                    "default": False,
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py').",
                    "default": "*",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = root


def tool_function(
    pattern: str,
    path: str,
    recursive: bool = True,
    case_sensitive: bool = False,
    file_pattern: str = "*",
) -> str:
    """Execute a search using grep."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = str(p.resolve())
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        # Build grep command
        cmd = ["grep"]
        
        # Add options
        if not case_sensitive:
            cmd.append("-i")  # Case insensitive
        if recursive:
            cmd.append("-r")
        cmd.append("-n")  # Show line numbers
        cmd.append("-H")  # Show filenames
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(p))
        
        # Add file pattern if specified and recursive
        if recursive and file_pattern != "*":
            cmd.extend(["--include", file_pattern])

        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # grep returns exit code 1 when no matches found
        if result.returncode == 1:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        # Truncate output if too long
        output = result.stdout
        max_len = 10000
        if len(output) > max_len:
            lines = output.split("\n")
            truncated = "\n".join(lines[:100])
            truncated += f"\n... ({len(lines) - 100} more lines)"
            output = truncated

        match_count = len([l for l in output.split("\n") if l.strip()])
        return f"Found {match_count} matches:\n{output}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern."
    except Exception as e:
        return f"Error: {e}"
