"""
File search tool: search for files by content patterns with advanced filtering.

Extends the basic search tool with content-based file discovery,
useful for finding files containing specific code patterns.
"""

from __future__ import annotations

import fnmatch
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _should_skip_file(path: Path, exclude_patterns: list[str] | None = None) -> bool:
    """Check if a file should be skipped based on patterns."""
    # Skip hidden files and directories
    if any(part.startswith('.') for part in path.parts):
        return True
    
    # Skip common non-code directories
    skip_dirs = {'__pycache__', 'node_modules', '.git', '.hg', '.svn', 'venv', '.venv', 'env'}
    if any(part in skip_dirs for part in path.parts):
        return True
    
    # Skip binary and generated files
    skip_extensions = {'.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.bin', 
                       '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.tar', 
                       '.gz', '.bz2', '.7z', '.rar', '.ico', '.woff', '.woff2',
                       '.ttf', '.eot', '.otf', '.svg', '.mp3', '.mp4', '.avi',
                       '.mov', '.webm', '.wasm', '.class', '.o', '.a', '.lib'}
    if path.suffix.lower() in skip_extensions:
        return True
    
    # Apply custom exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern):
                return True
    
    return False


def _search_single_file(
    path: Path,
    root: Path,
    compiled_pattern: re.Pattern,
    max_context_chars: int = 200,
) -> dict[str, Any] | None:
    """Search a single file for pattern matches."""
    try:
        # Skip files > 1MB
        if path.stat().st_size > 1_000_000:
            return None
        
        content = path.read_text(encoding="utf-8", errors="ignore")
        matches = list(compiled_pattern.finditer(content))
        
        if not matches:
            return None
        
        # Get context around first match
        first_match = matches[0]
        start = max(0, first_match.start() - 100)
        end = min(len(content), first_match.end() + 100)
        context = content[start:end].replace("\n", " ")
        
        return {
            "path": str(path.relative_to(root)),
            "match_count": len(matches),
            "context": context[:max_context_chars] + "..." if len(context) > max_context_chars else context,
        }
    except (OSError, UnicodeDecodeError):
        return None


def _search_files_by_content(
    directory: str,
    pattern: str,
    file_extensions: list[str] | None = None,
    max_results: int = 20,
    case_sensitive: bool = False,
    exclude_patterns: list[str] | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Search for files containing a regex pattern in their content.
    
    Args:
        directory: Root directory to search
        pattern: Regex pattern to search for
        file_extensions: List of file extensions to include (e.g., ['.py', '.js'])
        max_results: Maximum number of results to return
        case_sensitive: Whether the search is case sensitive
        exclude_patterns: List of glob patterns to exclude (e.g., ['*.min.js', 'test_*'])
        max_workers: Number of parallel workers for file processing
    
    Returns:
        Dictionary with search results and metadata
    """
    try:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return {"error": f"Directory not found: {directory}"}
        
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex pattern: {e}"}
        
        # Collect all files to search
        files_to_search = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            
            # Filter by extension if specified
            if file_extensions and path.suffix not in file_extensions:
                continue
            
            # Skip unwanted files
            if _should_skip_file(path, exclude_patterns):
                continue
            
            files_to_search.append(path)
        
        # Search files in parallel
        results = []
        files_searched = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(_search_single_file, path, root, compiled_pattern): path
                for path in files_to_search
            }
            
            for future in as_completed(future_to_path):
                files_searched += 1
                result = future.result()
                if result:
                    results.append(result)
                    if len(results) >= max_results:
                        # Cancel remaining futures
                        for f in future_to_path:
                            f.cancel()
                        break
        
        # Sort results by path for consistent output
        results.sort(key=lambda r: r["path"])
        
        return {
            "files_searched": files_searched,
            "matches_found": len(results),
            "results": results,
        }
        
    except Exception as e:
        return {"error": f"Search failed: {e}"}


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "file_search",
        "description": "Search for files containing specific content patterns with parallel processing and advanced filtering. Useful for finding code files that contain particular functions, classes, or text patterns. Automatically skips binary files, hidden files, and common non-code directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Root directory to search (absolute or relative path)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for in file contents",
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of file extensions to include (e.g., ['.py', '.js']). If not provided, searches all files.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching files to return (default: 20)",
                    "default": 20,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: false)",
                    "default": False,
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of glob patterns to exclude (e.g., ['*.min.js', 'test_*']). Hidden files and common non-code directories are automatically excluded.",
                },
            },
            "required": ["directory", "pattern"],
        },
    }


def tool_function(
    directory: str,
    pattern: str,
    file_extensions: list[str] | None = None,
    max_results: int = 20,
    case_sensitive: bool = False,
    exclude_patterns: list[str] | None = None,
) -> str:
    """Execute the file search tool."""
    result = _search_files_by_content(
        directory=directory,
        pattern=pattern,
        file_extensions=file_extensions,
        max_results=max_results,
        case_sensitive=case_sensitive,
        exclude_patterns=exclude_patterns,
    )
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    lines = [
        f"Searched {result['files_searched']} files, found {result['matches_found']} matches:",
        "",
    ]
    
    for i, match in enumerate(result['results'], 1):
        lines.append(f"{i}. {match['path']} ({match['match_count']} matches)")
        lines.append(f"   Context: {match['context']}")
        lines.append("")
    
    return "\n".join(lines)
