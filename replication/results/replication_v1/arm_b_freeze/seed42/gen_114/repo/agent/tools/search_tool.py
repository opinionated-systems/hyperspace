"""
Search tool: search for files and content within the codebase.

Provides grep-like and find-like functionality for exploring codebases.
"""

from __future__ import annotations

import subprocess
import os


def tool_info() -> dict:
    return {
        "name": "search",
        "description": "Search for files and content within a directory. Supports: (1) grep_search: search file contents using regex patterns, (2) find_files: find files by name pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep_search", "find_files"],
                    "description": "Type of search to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for grep, glob for find)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                    "default": 50,
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute search command."""
    if not os.path.isdir(path):
        return f"Error: Directory not found: {path}"

    try:
        if command == "grep_search":
            return _grep_search(path, pattern, file_extension, max_results)
        elif command == "find_files":
            return _find_files(path, pattern, max_results)
        else:
            return f"Error: Unknown command: {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep_search(
    path: str, pattern: str, file_extension: str | None, max_results: int
) -> str:
    """Search file contents using grep."""
    cmd = ["grep", "-r", "-n", "-I", "--include=*" + (file_extension or "")]
    
    # Add exclude patterns for common non-source directories
    cmd.extend(["--exclude-dir=.git", "--exclude-dir=__pycache__", "--exclude-dir=node_modules"])
    
    cmd.extend([pattern, path])
    
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30
    )
    
    lines = result.stdout.strip().split("\n") if result.stdout else []
    lines = [line for line in lines if line]
    
    if not lines:
        return "No matches found."
    
    if len(lines) > max_results:
        lines = lines[:max_results]
        lines.append(f"\n... ({len(lines)} total matches, showing first {max_results})")
    
    return "\n".join(lines)


def _find_files(path: str, pattern: str, max_results: int) -> str:
    """Find files by name pattern."""
    cmd = [
        "find", path,
        "-name", pattern,
        "-type", "f",
        "!", "-path", "*/.git/*",
        "!", "-path", "*/__pycache__/*",
        "!", "-path", "*/node_modules/*",
    ]
    
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30
    )
    
    files = result.stdout.strip().split("\n") if result.stdout else []
    files = [f for f in files if f]
    
    if not files:
        return "No files found."
    
    if len(files) > max_results:
        files = files[:max_results]
        files.append(f"\n... ({len(files)} total files, showing first {max_results})")
    
    return "\n".join(files)
