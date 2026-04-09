"""
Search tool: search for files and content within the codebase.

Provides grep-like functionality to find files by name pattern and
search for content within files. Useful for code discovery and navigation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content within the codebase. "
            "Supports finding files by pattern and searching file contents. "
            "Useful for discovering code locations and understanding the structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (file pattern or content to search).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: allowed root).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["files", "content"],
                    "description": "Type of search: 'files' for file names, 'content' for file contents.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter by when searching content (e.g., '*.py').",
                },
            },
            "required": ["query", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate(content: str, max_len: int = 5000) -> str:
    """Truncate long output to prevent context overflow."""
    if len(content) > max_len:
        lines = content.split("\n")
        # Keep first half and last half of lines
        half_lines = len(lines) // 2
        return "\n".join(lines[:half_lines]) + f"\n... [{len(lines) - half_lines} more lines] ...\n"
    return content


def tool_function(
    query: str,
    search_type: str,
    path: str | None = None,
    file_pattern: str | None = None,
) -> str:
    """Execute a search command.
    
    Args:
        query: Search query string
        search_type: Either 'files' or 'content'
        path: Directory to search in (defaults to allowed root)
        file_pattern: File pattern filter for content searches
        
    Returns:
        Search results as formatted string
    """
    try:
        # Validate inputs
        if not query:
            return "Error: query is required"
        
        if not search_type:
            return "Error: search_type is required (use 'files' or 'content')"
        
        # Determine search path
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        search_path = os.path.abspath(search_path)
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        # Validate search_type
        valid_types = ["files", "content"]
        if search_type not in valid_types:
            return f"Error: unknown search_type '{search_type}'. Use 'files' or 'content'."
        
        if search_type == "files":
            return _search_files(query, search_path)
        else:  # search_type == "content"
            return _search_content(query, search_path, file_pattern)
            
    except Exception as e:
        import traceback
        return f"Error: {e}\nTraceback: {traceback.format_exc()}"


def _search_files(pattern: str, search_path: str) -> str:
    """Search for files by name pattern."""
    try:
        # Use find command for file search
        cmd = [
            "find", search_path,
            "-type", "f",
            "-name", pattern,
            "-not", "-path", "*/\.*",  # Exclude hidden files
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error searching files: {result.stderr}"
        
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        
        if not files or files == ['']:
            return f"No files matching '{pattern}' found in {search_path}"
        
        # Format results
        output = f"Found {len(files)} file(s) matching '{pattern}':\n"
        for f in files[:50]:  # Limit to 50 results
            if f:
                # Show relative path if under allowed root
                if _ALLOWED_ROOT and f.startswith(_ALLOWED_ROOT):
                    rel_path = f[len(_ALLOWED_ROOT):].lstrip("/")
                    output += f"  {rel_path}\n"
                else:
                    output += f"  {f}\n"
        
        if len(files) > 50:
            output += f"  ... and {len(files) - 50} more files\n"
        
        return _truncate(output)
        
    except subprocess.TimeoutExpired:
        return "Error: search timed out (took longer than 30 seconds)"
    except Exception as e:
        return f"Error searching files: {e}"


def _search_content(query: str, search_path: str, file_pattern: str | None = None) -> str:
    """Search for content within files using grep."""
    try:
        # Build grep command
        cmd = [
            "grep", "-r",
            "-n",  # Show line numbers
            "-i",  # Case insensitive
            "--include", file_pattern or "*",
            query,
            search_path,
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in [0, 1]:
            return f"Error searching content: {result.stderr}"
        
        if not result.stdout.strip():
            return f"No matches for '{query}' found in {search_path}"
        
        # Format results
        lines = result.stdout.strip().split("\n")
        output = f"Found {len(lines)} match(es) for '{query}':\n\n"
        
        for line in lines[:30]:  # Limit to 30 matches
            if line:
                # Parse grep output: path:line:content
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path, line_num, content = parts[0], parts[1], parts[2]
                    # Show relative path
                    if _ALLOWED_ROOT and file_path.startswith(_ALLOWED_ROOT):
                        rel_path = file_path[len(_ALLOWED_ROOT):].lstrip("/")
                        output += f"{rel_path}:{line_num}: {content.strip()}\n"
                    else:
                        output += f"{file_path}:{line_num}: {content.strip()}\n"
        
        if len(lines) > 30:
            output += f"\n... and {len(lines) - 30} more matches\n"
        
        return _truncate(output, 8000)
        
    except subprocess.TimeoutExpired:
        return "Error: search timed out (took longer than 30 seconds)"
    except Exception as e:
        return f"Error searching content: {e}"
