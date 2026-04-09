"""
Search tool: find patterns in files using grep/ripgrep.

Provides code search capabilities to help locate files and patterns
during codebase modification.
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
            "Returns matching lines with file paths and line numbers. "
            "Useful for finding code patterns, function definitions, etc."
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
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
) -> str:
    """Search for pattern in files. Returns matching lines."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        # Validate pattern is not empty
        if not pattern or not pattern.strip():
            return "Error: pattern cannot be empty"

        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include", file_extension or "*", pattern, str(p)]
        
        # If ripgrep is available, use it (faster, better output)
        try:
            # Add file extension filter if provided
            rg_cmd = ["rg", "-n", "-I"]
            if file_extension:
                rg_cmd.extend(["-g", f"*{file_extension}"])
            rg_cmd.extend([pattern, str(p)])
            
            result = subprocess.run(
                rg_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fall back to grep
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Filter out empty lines
            lines = [line for line in lines if line.strip()]
            # Limit output
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return "\n".join(lines) if lines else "No matches found."
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Search error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"
