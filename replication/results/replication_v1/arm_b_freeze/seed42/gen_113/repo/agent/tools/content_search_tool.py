"""
Content search tool: search for files containing specific text patterns.

This tool extends the search capabilities by allowing content-based searches
across multiple files, useful for finding code patterns or references.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _should_ignore(path: str, ignore_patterns: list[str]) -> bool:
    """Check if a path should be ignored based on patterns."""
    name = os.path.basename(path)
    for pattern in ignore_patterns:
        # Match against the basename (filename/directory name)
        if fnmatch.fnmatch(name, pattern):
            return True
        # Match against the full path, but only if pattern contains path separators
        # or is not a simple hidden file pattern (.*)
        if '/' in pattern or pattern != ".*":
            if fnmatch.fnmatch(path, pattern):
                return True
    return False


def _search_in_file(file_path: str, pattern: str, case_sensitive: bool = False) -> list[dict]:
    """Search for pattern in a single file and return matches with context."""
    matches = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except (IOError, OSError, UnicodeDecodeError) as e:
        logger.debug(f"Could not read {file_path}: {e}")
        return matches
    
    pattern_lower = pattern.lower() if not case_sensitive else pattern
    
    for line_num, line in enumerate(lines, 1):
        line_compare = line if case_sensitive else line.lower()
        if pattern_lower in line_compare:
            # Get context (2 lines before and after)
            start = max(0, line_num - 3)
            end = min(len(lines), line_num + 2)
            context = ''.join(lines[start:end])
            
            matches.append({
                "line": line_num,
                "content": line.rstrip(),
                "context": context,
            })
    
    return matches


def content_search(
    path: str,
    pattern: str,
    file_pattern: str = "*",
    case_sensitive: bool = False,
    max_results: int = 50,
    include_hidden: bool = False,
) -> dict[str, Any]:
    """Search for files containing specific text patterns.

    Args:
        path: Directory path to search in.
        pattern: Text pattern to search for within files.
        file_pattern: Glob pattern for files to include (default: "*" for all).
        case_sensitive: Whether to perform case-sensitive search.
        max_results: Maximum number of file matches to return.
        include_hidden: Whether to include hidden files and directories.

    Returns:
        Dictionary with search results containing file paths and match details.
    """
    if not os.path.isdir(path):
        return {
            "error": f"Path is not a directory: {path}",
            "matches": [],
            "total_files_searched": 0,
            "total_matches": 0,
        }

    ignore_patterns = [
        ".git", ".svn", ".hg",  # Version control
        "__pycache__", ".pytest_cache", ".mypy_cache",  # Python cache
        "node_modules", "vendor",  # Dependencies
        "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll",  # Compiled
        ".env", ".venv", "venv", "env",  # Virtual environments
        ".idea", ".vscode",  # IDE
        "*.min.js", "*.min.css",  # Minified files
        "*.log",  # Log files
    ]
    
    if not include_hidden:
        ignore_patterns.append(".*")

    matches = []
    files_searched = 0
    total_match_count = 0

    try:
        for root, dirs, files in os.walk(path):
            # Filter out ignored directories
            dirs[:] = [
                d for d in dirs 
                if not _should_ignore(os.path.join(root, d), ignore_patterns)
            ]
            
            for filename in files:
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue
                    
                file_path = os.path.join(root, filename)
                
                if _should_ignore(file_path, ignore_patterns):
                    continue
                
                files_searched += 1
                file_matches = _search_in_file(file_path, pattern, case_sensitive)
                
                if file_matches:
                    total_match_count += len(file_matches)
                    matches.append({
                        "file": file_path,
                        "matches": file_matches[:10],  # Limit matches per file
                        "match_count": len(file_matches),
                    })
                    
                    if len(matches) >= max_results:
                        return {
                            "matches": matches,
                            "total_files_searched": files_searched,
                            "total_matches": total_match_count,
                            "truncated": True,
                            "pattern": pattern,
                        }
    except Exception as e:
        logger.error(f"Error during content search: {e}")
        return {
            "error": str(e),
            "matches": matches,
            "total_files_searched": files_searched,
            "total_matches": total_match_count,
        }

    return {
        "matches": matches,
        "total_files_searched": files_searched,
        "total_matches": total_match_count,
        "pattern": pattern,
    }


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "content_search",
        "description": (
            "Search for files containing specific text patterns. "
            "Returns file paths with line numbers and context for each match. "
            "Useful for finding code references, function definitions, or specific text across multiple files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search in",
                },
                "pattern": {
                    "type": "string",
                    "description": "Text pattern to search for within files",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to include (e.g., '*.py', '*.js'). Default: '*' for all files",
                    "default": "*",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether to perform case-sensitive search. Default: false",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of file matches to return. Default: 50",
                    "default": 50,
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files and directories. Default: false",
                    "default": False,
                },
            },
            "required": ["path", "pattern"],
        },
    }


tool_function = content_search
