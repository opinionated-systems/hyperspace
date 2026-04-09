"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to find code patterns, text, or references.
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
            "Supports regex patterns, file filtering, and directory scoping. "
            "Useful for finding code references, TODOs, or specific patterns."
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
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter by (e.g., '*.py', '*.js'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive. Default: true.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
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


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n... [output truncated] ...\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Execute a search for pattern in files."""
    try:
        # Validate inputs
        if not pattern:
            return "Error: pattern is required."
        if not path:
            return "Error: path is required."

        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path. Please provide an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}. The path {resolved} is outside the allowed scope."

        # Build grep command
        cmd_parts = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd_parts.append("-i")
        
        # Limit results
        cmd_parts.extend(["-m", str(max_results)])
        
        # Add pattern
        cmd_parts.append(pattern)
        
        # Add path
        cmd_parts.append(str(p))
        
        # Add file pattern if specified
        if file_pattern:
            cmd_parts.extend(["--include", file_pattern])
        
        # Exclude hidden directories
        cmd_parts.extend(["--exclude-dir", ".*"])
        
        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            # grep returns 1 when no matches found, which is not an error
            if result.returncode not in (0, 1):
                return f"Error running search: {result.stderr}"
            
            output = result.stdout.strip()
            if not output:
                return f"No matches found for pattern '{pattern}' in {path}"
            
            return _truncate(output)
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30s. Try a more specific pattern or narrower scope."
        except Exception as e:
            return f"Error running search: {e}"
            
    except Exception as e:
        return f"Error: {e}"
