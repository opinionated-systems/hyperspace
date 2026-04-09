"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality and file discovery to help the agent
navigate and understand the codebase structure.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for files by name pattern or search file contents with regex. Helps navigate and understand the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep"],
                    "description": "Search command to execute",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (relative or absolute)",
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern (for find_files) or regex pattern (for grep)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file glob pattern to limit grep search (e.g., '*.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


def _find_files(base_path: str, pattern: str, max_results: int = 50) -> list[str]:
    """Find files matching the given glob pattern."""
    results = []
    base = Path(base_path).expanduser().resolve()
    
    if not base.exists():
        return [f"Error: Path '{base_path}' does not exist"]
    
    try:
        for path in base.rglob(pattern):
            if path.is_file():
                results.append(str(path.relative_to(base)))
                if len(results) >= max_results:
                    results.append(f"... (truncated at {max_results} results)")
                    break
    except Exception as e:
        return [f"Error searching files: {e}"]
    
    return results if results else [f"No files matching '{pattern}' found in '{base_path}'"]


def _grep_files(base_path: str, pattern: str, file_pattern: str | None = None, max_results: int = 50) -> list[dict]:
    """Search file contents with regex pattern."""
    results = []
    base = Path(base_path).expanduser().resolve()
    
    if not base.exists():
        return [{"error": f"Path '{base_path}' does not exist"}]
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return [{"error": f"Invalid regex pattern: {e}"}]
    
    files_searched = 0
    matches_found = 0
    
    try:
        if file_pattern:
            files_to_search = list(base.rglob(file_pattern))
        else:
            files_to_search = [p for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")]
        
        for file_path in files_to_search:
            if not file_path.is_file():
                continue
            
            # Skip binary files and very large files
            try:
                stat = file_path.stat()
                if stat.st_size > 1024 * 1024:  # Skip files > 1MB
                    continue
            except OSError:
                continue
            
            files_searched += 1
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append({
                                "file": str(file_path.relative_to(base)),
                                "line": line_num,
                                "content": line.rstrip()[:200],  # Limit line length
                            })
                            matches_found += 1
                            if matches_found >= max_results:
                                results.append({
                                    "info": f"... (truncated at {max_results} matches)"
                                })
                                return results
            except (IOError, OSError, UnicodeDecodeError):
                continue
                
    except Exception as e:
        return [{"error": f"Error during grep: {e}"}]
    
    if not results:
        return [{"info": f"No matches for pattern '{pattern}' in '{base_path}' (searched {files_searched} files)"}]
    
    return results


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute search command.
    
    Args:
        command: Either 'find_files' or 'grep'
        path: Directory path to search in
        pattern: File pattern (find_files) or regex pattern (grep)
        file_pattern: Optional file glob to limit grep search
        max_results: Maximum results to return
    
    Returns:
        JSON string with search results
    """
    import json
    
    if command == "find_files":
        results = _find_files(path, pattern, max_results)
        return json.dumps({"files": results}, indent=2)
    
    elif command == "grep":
        results = _grep_files(path, pattern, file_pattern, max_results)
        return json.dumps({"matches": results}, indent=2)
    
    else:
        return json.dumps({"error": f"Unknown command: {command}"})
