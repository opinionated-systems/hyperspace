"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports searching file contents and file names."
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
                    "description": "Directory to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py').",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Whether to search in file contents or file names.",
                },
            },
            "required": ["pattern", "path"],
        },
    }


def _truncate(content: str, max_len: int = 5000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    search_type: str = "content",
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        if not p.exists():
            return f"Error: {path} does not exist."
        if not p.is_dir():
            return f"Error: {path} is not a directory."

        if search_type == "filename":
            # Search for files by name
            cmd = ["find", str(p), "-type", "f", "-name", f"*{pattern}*"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            files = [f for f in result.stdout.strip().split("\n") if f]
            if not files:
                return f"No files matching '{pattern}' found in {path}"
            return f"Files matching '{pattern}':\n" + "\n".join(files[:50])

        else:
            # Search in file contents using grep
            cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", pattern, str(p)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0 and not result.stdout:
                return f"No matches found for '{pattern}' in {path}"
            
            lines = result.stdout.strip().split("\n")
            lines = [l for l in lines if l]
            
            if not lines:
                return f"No matches found for '{pattern}' in {path}"
            
            output = "\n".join(lines[:100])  # Limit results
            if len(lines) > 100:
                output += f"\n... and {len(lines) - 100} more matches"
            
            return _truncate(f"Matches for '{pattern}':\n{output}")

    except Exception as e:
        return f"Error: {e}"
