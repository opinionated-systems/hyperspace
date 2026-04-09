"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore codebases.
Enhanced with context lines, multiple file extensions, and file listing.
"""

from __future__ import annotations

import subprocess
import os
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. Returns matching lines with file paths and line numbers. "
            "Also supports listing files in directories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for (for search command)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js')",
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file extensions to include (e.g., ['.py', '.js'])",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: True)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before/after matches (default: 0)",
                },
                "command": {
                    "type": "string",
                    "enum": ["search", "list_files"],
                    "description": "Command to execute: 'search' for content, 'list_files' for directory listing",
                    "default": "search",
                },
            },
            "required": [],
        },
    }


def tool_function(
    pattern: str = "",
    path: str = ".",
    file_extension: str | None = None,
    file_extensions: list[str] | None = None,
    case_sensitive: bool = True,
    max_results: int = 50,
    context_lines: int = 0,
    command: str = "search",
) -> str:
    """Search for a pattern in files or list directory contents.

    Args:
        pattern: The regex pattern to search for (for search command)
        path: Directory or file to search in
        file_extension: Optional single file extension filter
        file_extensions: Optional list of file extensions to include
        case_sensitive: Whether the search is case sensitive
        max_results: Maximum number of results to return
        context_lines: Number of context lines to show before/after matches
        command: The command to execute ('search' or 'list_files')

    Returns:
        Matching lines with file paths and line numbers, or directory listing
    """
    if command == "list_files":
        return _list_files(path, max_results)
    
    return _search_content(
        pattern=pattern,
        path=path,
        file_extension=file_extension,
        file_extensions=file_extensions,
        case_sensitive=case_sensitive,
        max_results=max_results,
        context_lines=context_lines,
    )


def _search_content(
    pattern: str,
    path: str,
    file_extension: str | None,
    file_extensions: list[str] | None,
    case_sensitive: bool,
    max_results: int,
    context_lines: int,
) -> str:
    """Search for pattern in file contents."""
    if not pattern:
        return "Error: pattern is required for search command"
    
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]

        if not case_sensitive:
            cmd.append("-i")
        
        # Add context lines if specified
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        # Add pattern
        cmd.append(pattern)

        # Add path
        cmd.append(path)

        # Add file extension filter(s)
        if file_extensions:
            for ext in file_extensions:
                cmd.extend(["--include", f"*{ext}"])
        elif file_extension:
            cmd.extend(["--include", f"*{file_extension}"])

        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # grep returns exit code 1 when no matches found
        if result.returncode not in (0, 1):
            return f"Error: grep failed with exit code {result.returncode}: {result.stderr}"

        lines = result.stdout.strip().split("\n") if result.stdout else []

        if not lines or lines == [""]:
            return f"No matches found for pattern '{pattern}'"

        # Limit results
        total_matches = len([l for l in lines if l and not l.startswith("--")])
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({total_matches - max_results} more results)")

        return f"Found {total_matches} matches:\n" + "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def _list_files(path: str, max_results: int = 200) -> str:
    """List files in a directory."""
    try:
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist"
        
        if os.path.isfile(path):
            return f"{path} (file)"
        
        # Use find to list files
        result = subprocess.run(
            ["find", path, "-maxdepth", "2", "-not", "-path", "*/\.*"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode != 0:
            return f"Error: find failed: {result.stderr}"
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        if not lines or lines == [""]:
            return f"No files found in '{path}'"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more items)")
        
        return f"Contents of '{path}':\n" + "\n".join(lines)
        
    except subprocess.TimeoutExpired:
        return "Error: Listing timed out after 10 seconds"
    except Exception as e:
        return f"Error: {e}"
