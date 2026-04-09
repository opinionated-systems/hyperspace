"""
Search tool: grep and find functionality for searching file contents.

Provides file search capabilities to help agents locate code patterns,
function definitions, and specific text within the codebase.
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
            "Search for patterns in files using grep-like functionality. "
            "Can search for text patterns, function definitions, or specific file types. "
            "Results are limited to avoid overwhelming output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex or plain text).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path). Default: allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', '*.js'). Default: all files.",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Whether pattern is a regex (true) or plain text (false). Default: false.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None
_MAX_OUTPUT_LEN = 10000


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_allowed(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def _truncate_output(content: str, max_len: int = _MAX_OUTPUT_LEN) -> str:
    """Truncate output if too long, keeping beginning and end."""
    if len(content) <= max_len:
        return content
    
    lines = content.split("\n")
    total_lines = len(lines)
    
    # Keep roughly half of max_len from start and half from end
    # Approximately 50 lines total (25 from start, 25 from end)
    keep_lines = 50
    half_keep = keep_lines // 2
    
    if total_lines <= keep_lines:
        # If not many lines, truncate by character instead
        half_len = max_len // 2
        return content[:half_len] + f"\n... [{len(content) - max_len} chars truncated] ...\n" + content[-half_len:]
    
    truncated_count = total_lines - keep_lines
    return (
        "\n".join(lines[:half_keep]) +
        f"\n... [{truncated_count} lines truncated] ...\n" +
        "\n".join(lines[-half_keep:])
    )


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    is_regex: bool = False,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        # Determine search path
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        
        if not _is_within_allowed(search_path):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if not os.path.exists(search_path):
            return f"Error: path does not exist: {search_path}"
        
        results = []
        count = 0
        
        # Build file list
        if os.path.isfile(search_path):
            files_to_search = [search_path]
        else:
            # Find files matching pattern
            if file_pattern:
                files_to_search = list(Path(search_path).rglob(file_pattern))
                # Filter out hidden directories and __pycache__
                files_to_search = [
                    f for f in files_to_search 
                    if not any(part.startswith(".") or part == "__pycache__" 
                              for part in f.parts)
                ]
            else:
                files_to_search = [
                    f for f in Path(search_path).rglob("*")
                    if f.is_file() and not any(
                        part.startswith(".") or part == "__pycache__"
                        for part in f.parts
                    )
                ]
        
        # Compile regex if needed
        if is_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return f"Error: invalid regex pattern: {e}"
        else:
            # Escape special regex characters for plain text search
            escaped = re.escape(pattern)
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(escaped, flags)
        
        # Search files
        for file_path in files_to_search:
            if count >= max_results:
                break
                
            try:
                # Skip binary files
                if _is_binary(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            rel_path = os.path.relpath(file_path, search_path)
                            results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                            count += 1
                            if count >= max_results:
                                break
            except (IOError, OSError, PermissionError):
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}'"
        
        output = f"Found {count} match(es) for pattern '{pattern}':\n" + "\n".join(results)
        if count >= max_results:
            output += f"\n... (results limited to {max_results} matches)"
        
        return _truncate_output(output)
        
    except Exception as e:
        return f"Error during search: {e}"


def _is_binary(file_path: Path) -> bool:
    """Check if file is binary by reading first chunk."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(8192)  # Read more bytes for better detection
            # Check for null bytes or high ratio of non-printable bytes
            if b'\x00' in chunk:
                return True
            # Check if more than 30% of bytes are non-printable (excluding common whitespace)
            non_printable = sum(1 for b in chunk if b < 32 and b not in (9, 10, 13))
            if len(chunk) > 0 and non_printable / len(chunk) > 0.3:
                return True
            return False
    except (IOError, OSError, PermissionError):
        return True
