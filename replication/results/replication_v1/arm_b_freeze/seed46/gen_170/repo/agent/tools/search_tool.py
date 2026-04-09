"""
Search tool: find patterns in files using grep and find.

Provides structured search capabilities for the agent to locate
files and content within the codebase.
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
            "Can search file contents or find files by name. "
            "Results are truncated to prevent context overflow."
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
                    "description": "Directory to search in (absolute path). Default: allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to search (e.g., '*.py'). Default: all files.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Search in file contents or find files by name. Default: content.",
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
    file_pattern: str | None = None,
    search_type: str = "content",
) -> str:
    """Execute a search command."""
    try:
        # Determine search path
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        search_path = os.path.abspath(search_path)
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if not os.path.exists(search_path):
            return f"Error: path {search_path} does not exist."
        
        if search_type == "filename":
            # Find files by name pattern
            cmd = ["find", search_path, "-type", "f", "-name", pattern]
            # Exclude hidden directories
            cmd.extend(["-not", "-path", "*/\.*"])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout.strip()
            if not output:
                return f"No files matching '{pattern}' found in {search_path}."
            files = output.split("\n")[:50]  # Limit results
            return f"Found {len(files)} file(s) matching '{pattern}':\n" + "\n".join(files)
        
        else:
            # Search file contents with grep
            cmd = [
                "grep", "-r", "-n", "-I",
                "--include", file_pattern or "*",
                "-E", pattern,
                search_path,
            ]
            # Exclude hidden directories
            cmd.extend(["--exclude-dir", ".*"])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0 and not result.stdout:
                return f"No matches found for pattern '{pattern}' in {search_path}."
            
            # Process and truncate results
            lines = result.stdout.strip().split("\n")
            total_matches = len(lines)
            
            # Limit output to prevent context overflow
            max_lines = 100
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                truncated_msg = f"\n... ({total_matches - max_lines} more matches truncated) ..."
            else:
                truncated_msg = ""
            
            output = "\n".join(lines)
            return f"Found {total_matches} match(es) for '{pattern}':\n{output}{truncated_msg}"
    
    except subprocess.TimeoutExpired:
        return f"Error: search timed out after 30s. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error during search: {e}"
