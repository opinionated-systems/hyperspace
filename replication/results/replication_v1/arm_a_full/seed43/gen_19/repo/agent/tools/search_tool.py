"""
Search tool: search for text patterns in files.

Provides grep-like functionality to search for patterns across files,
with support for regex, file filtering, and context lines.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files. "
            "Supports regex patterns, file glob filtering, and context lines. "
            "Useful for finding code patterns, function definitions, or references."
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
                    "description": "File glob pattern to filter files (e.g., '*.py'). Optional.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show around matches. Default: 2.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: False.",
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
    file_pattern: str | None = None,
    context_lines: int = 2,
    case_sensitive: bool = False,
) -> str:
    """Search for pattern in files."""
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
            return f"Error: {p} does not exist."

        # Build grep command
        cmd_parts = ["grep", "-r", "-n"]
        
        # Add context lines
        if context_lines > 0:
            cmd_parts.extend([f"-C", str(context_lines)])
        
        # Case sensitivity
        if not case_sensitive:
            cmd_parts.append("-i")
        
        # Add pattern
        cmd_parts.append(pattern)
        
        # Add path
        cmd_parts.append(str(p))
        
        # Add file pattern filter if specified
        if file_pattern:
            # Use find with grep for file pattern filtering
            find_cmd = [
                "find", str(p), "-type", "f", "-name", file_pattern
            ]
            result = subprocess.run(
                find_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return f"No files matching pattern '{file_pattern}' found in {p}"
            
            files = result.stdout.strip().split("\n")
            if not files or files == ['']:
                return f"No files matching pattern '{file_pattern}' found in {p}"
            
            # Search in found files
            cmd_parts = ["grep", "-n"]
            if context_lines > 0:
                cmd_parts.extend([f"-C", str(context_lines)])
            if not case_sensitive:
                cmd_parts.append("-i")
            cmd_parts.append(pattern)
            cmd_parts.extend(files)
        
        # Run the search
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode == 0:
            output = result.stdout
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {p}"
        else:
            # Error
            return f"Search error: {result.stderr}"
        
        if not output.strip():
            return f"No matches found for pattern '{pattern}' in {p}"
        
        # Format output
        lines = output.strip().split("\n")
        formatted_lines = []
        for line in lines:
            if line.startswith("--"):
                formatted_lines.append(line)
            elif ":" in line:
                parts = line.split(":", 2)
                if len(parts) >= 2:
                    file_path = parts[0]
                    line_num = parts[1]
                    content = parts[2] if len(parts) > 2 else ""
                    formatted_lines.append(f"{file_path}:{line_num}:{content}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        result_text = "\n".join(formatted_lines)
        return _truncate(result_text, 8000)
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 60s. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error during search: {e}"
