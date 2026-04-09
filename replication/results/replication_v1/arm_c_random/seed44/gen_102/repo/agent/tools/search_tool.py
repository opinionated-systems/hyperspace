"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality and file finding capabilities
to help the agent navigate and explore codebases efficiently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search tool for finding files and searching content. "
            "Commands: find (find files by name pattern), grep (search content). "
            "Useful for exploring codebases and locating specific code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find", "grep"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename for find, regex for grep).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate(content: str, max_len: int = 5000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<output truncated>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_extension: str | None = None,
) -> str:
    """Execute a search command."""
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

        if command == "find":
            return _find(p, pattern, file_extension)
        elif command == "grep":
            return _grep(p, pattern, file_extension)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find(p: Path, pattern: str, file_extension: str | None) -> str:
    """Find files by name pattern."""
    if not p.is_dir():
        return f"Error: {p} is not a directory."

    # Use find command for efficient searching
    cmd = ["find", str(p), "-type", "f", "-name", f"*{pattern}*"]
    
    if file_extension:
        cmd = ["find", str(p), "-type", "f", "-name", f"*{pattern}*{file_extension}"]
    
    # Exclude hidden directories
    cmd.extend(["-not", "-path", "*/\.*"])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return f"Error running find: {result.stderr}"
    
    files = [f for f in result.stdout.strip().split("\n") if f]
    
    if not files:
        return f"No files found matching '{pattern}' in {p}"
    
    output = f"Found {len(files)} file(s) matching '{pattern}':\n"
    for f in files[:20]:  # Limit to first 20 results
        output += f"  {f}\n"
    
    if len(files) > 20:
        output += f"  ... and {len(files) - 20} more files\n"
    
    return _truncate(output)


def _grep(p: Path, pattern: str, file_extension: str | None) -> str:
    """Search file contents for pattern."""
    if not p.is_dir():
        return f"Error: {p} is not a directory."

    # Build grep command
    cmd = ["grep", "-r", "-n", "-i", pattern, str(p)]
    
    if file_extension:
        cmd.extend(["--include", f"*{file_extension}"])
    
    # Exclude hidden directories
    cmd.extend(["--exclude-dir", ".*"])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # grep returns 1 when no matches found, which is not an error
    if result.returncode not in [0, 1]:
        return f"Error running grep: {result.stderr}"
    
    lines = [l for l in result.stdout.strip().split("\n") if l]
    
    if not lines:
        return f"No matches found for '{pattern}' in {p}"
    
    output = f"Found {len(lines)} match(es) for '{pattern}':\n"
    for line in lines[:30]:  # Limit to first 30 matches
        output += f"  {line}\n"
    
    if len(lines) > 30:
        output += f"  ... and {len(lines) - 30} more matches\n"
    
    return _truncate(output)
