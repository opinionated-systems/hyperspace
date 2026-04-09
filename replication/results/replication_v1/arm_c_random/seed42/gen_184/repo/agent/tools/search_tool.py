"""
Search tool: find text patterns in files using grep-like functionality.

Provides content search capabilities to complement the file and editor tools.
Enhanced with context lines, file type filtering, and better result formatting.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata matching the paper's tool schema."""
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files. Supports regex and recursive directory search. "
            "Enhanced with context lines, file type filtering, and result statistics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Default: current directory",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in directories. Default: True",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search. Default: False",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before/after match. Default: 0",
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file extensions to include (e.g., ['.py', '.js']). Default: all",
                },
            },
            "required": ["pattern"],
        },
    }


# Binary file extensions to skip
_BINARY_EXTENSIONS = {
    '.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.dat',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
    '.mp3', '.mp4', '.avi', '.mov', '.wav', '.ogg',
    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.db', '.sqlite', '.sqlite3',
}


def _should_skip_file(filepath: str, file_extensions: list[str] | None) -> bool:
    """Check if a file should be skipped based on extension filters."""
    ext = os.path.splitext(filepath)[1].lower()
    
    # Always skip binary files
    if ext in _BINARY_EXTENSIONS:
        return True
    
    # Skip pycache
    if "__pycache__" in filepath or filepath.endswith(".pyc"):
        return True
    
    # Apply extension filter if specified
    if file_extensions:
        return ext not in [fe.lower() for fe in file_extensions]
    
    return False


def tool_function(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
    context_lines: int = 0,
    file_extensions: list[str] | None = None,
) -> str:
    """Search for text patterns in files.

    Args:
        pattern: The search pattern (regex supported)
        path: File or directory to search in
        recursive: Search recursively in directories
        case_sensitive: Case-sensitive search
        max_results: Maximum number of results to return
        context_lines: Number of context lines to show before/after match
        file_extensions: List of file extensions to include (e.g., ['.py', '.js'])

    Returns:
        Formatted search results or error message
    """
    if not pattern:
        return "Error: pattern is required"

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results = []
    result_count = 0
    files_searched = 0
    files_matched = 0

    if os.path.isfile(path):
        # Search single file
        files_to_search = [path]
    elif os.path.isdir(path):
        # Search directory
        if recursive:
            files_to_search = []
            for root, _, files in os.walk(path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    if not _should_skip_file(filepath, file_extensions):
                        files_to_search.append(filepath)
        else:
            files_to_search = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
                and not _should_skip_file(os.path.join(path, f), file_extensions)
            ]
    else:
        return f"Error: path '{path}' does not exist"

    for filepath in files_to_search:
        files_searched += 1
        file_has_match = False
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                if compiled_pattern.search(line):
                    if not file_has_match:
                        file_has_match = True
                        files_matched += 1
                    
                    # Build result with context
                    if context_lines > 0:
                        start_ctx = max(0, line_num - context_lines - 1)
                        end_ctx = min(len(lines), line_num + context_lines)
                        
                        result_parts = [f"{filepath}:{line_num}:"]
                        for ctx_line_num in range(start_ctx, end_ctx):
                            prefix = ">>> " if ctx_line_num == line_num - 1 else "    "
                            result_parts.append(f"{prefix}{ctx_line_num + 1:4d}: {lines[ctx_line_num].rstrip()}")
                        results.append("\n".join(result_parts))
                    else:
                        results.append(f"{filepath}:{line_num}:{line.rstrip()}")
                    
                    result_count += 1
                    if result_count >= max_results:
                        break
            if result_count >= max_results:
                break
        except Exception as e:
            continue

    if not results:
        filter_info = f" (filtered to {file_extensions})" if file_extensions else ""
        return f"No matches found for pattern '{pattern}' in '{path}'{filter_info}\n[Searched {files_searched} files]"

    # Build output with statistics
    output_parts = [
        f"Search Results for '{pattern}':",
        f"  Files searched: {files_searched}",
        f"  Files with matches: {files_matched}",
        f"  Total matches: {result_count}{'+' if result_count >= max_results else ''}",
        "",
        "Matches:",
    ]
    output_parts.extend(results)
    
    if result_count >= max_results:
        output_parts.append(f"\n... (truncated to {max_results} results)")

    return "\n".join(output_parts)
