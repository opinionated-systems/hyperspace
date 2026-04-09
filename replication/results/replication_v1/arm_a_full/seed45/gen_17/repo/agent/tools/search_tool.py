"""
Search tool: find patterns in files using grep and recursive search.

Provides structured search capabilities for the meta agent to locate
code patterns, function definitions, and text across the codebase.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep-like functionality. "
            "Supports regex patterns, file filtering, and recursive directory search. "
            "Useful for finding code patterns, function definitions, and text."
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
                    "description": "Absolute path to file or directory to search in.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py', '*.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: false).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
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


def _truncate_output(output: str, max_chars: int = 10000) -> str:
    """Truncate output if it exceeds max length."""
    if len(output) > max_chars:
        return output[: max_chars // 2] + "\n... [output truncated] ...\n" + output[-max_chars // 2 :]
    return output


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search for patterns in files."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        # Build grep command as a list (safer than shell string)
        cmd_parts = ["grep", "-r", "-n"]
        
        # Case sensitivity
        if not case_sensitive:
            cmd_parts.append("-i")
        
        # File pattern filter
        if file_pattern:
            cmd_parts.extend(["--include", file_pattern])
        
        # Add pattern and path
        cmd_parts.extend([pattern, str(p)])
        
        # Execute search
        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # grep returns 1 when no matches found, which is not an error
            if result.returncode not in [0, 1]:
                return f"Error: Search failed with return code {result.returncode}: {result.stderr}"
            
            output = result.stdout.strip()
            
            if not output:
                return f"No matches found for pattern '{pattern}' in {path}"
            
            # Split into lines and limit results
            lines = output.split("\n")
            if len(lines) > max_results:
                lines = lines[:max_results]
                truncated = True
            else:
                truncated = False
            
            header = f"Found matches for pattern '{pattern}' in {path}:"
            if file_pattern:
                header += f" (filtered to {file_pattern})"
            if truncated:
                header += f"\nShowing first {max_results} of {len(output.split(chr(10)))} total matches:\n"
            else:
                header += f"\nShowing all {len(lines)} matches:\n"
            
            return header + _truncate_output("\n".join(lines))
            
        except subprocess.TimeoutExpired:
            return f"Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
        except Exception as e:
            return f"Error executing search: {type(e).__name__}: {e}"
            
    except Exception as e:
        return f"Error: {e}"
