"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
Supports pattern matching, file filtering, and directory scoping.
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
            "Supports regex patterns, file extension filtering, and directory scoping. "
            "Useful for finding code references, function definitions, etc."
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
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
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


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search for pattern in files."""
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
            return f"Error: {p} does not exist."

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
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode not in (0, 1):  # 0 = matches found, 1 = no matches
            return f"Error: grep failed with code {result.returncode}: {result.stderr}"
        
        if not result.stdout.strip():
            return f"No matches found for pattern '{pattern}' in {p}"
        
        # Process and limit results
        lines = result.stdout.strip().split("\n")
        total_matches = len(lines)
        
        if total_matches > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total_matches - max_results} more results truncated)"
        else:
            truncated_msg = ""
        
        output = "\n".join(lines) + truncated_msg
        return _truncate(f"Found {total_matches} match(es) for '{pattern}':\n{output}")
        
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error: {e}"
