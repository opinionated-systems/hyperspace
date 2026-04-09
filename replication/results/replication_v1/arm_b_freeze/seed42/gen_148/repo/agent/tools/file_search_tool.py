"""
File search tool: search for files by content patterns with advanced filtering.

Extends the basic search tool with content-based file discovery,
useful for finding files containing specific code patterns.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _search_files_by_content(
    directory: str,
    pattern: str,
    file_extensions: list[str] | None = None,
    max_results: int = 20,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Search for files containing a regex pattern in their content.
    
    Args:
        directory: Root directory to search
        pattern: Regex pattern to search for
        file_extensions: List of file extensions to include (e.g., ['.py', '.js'])
        max_results: Maximum number of results to return
        case_sensitive: Whether the search is case sensitive
    
    Returns:
        Dictionary with search results and metadata
    """
    try:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return {"error": f"Directory not found: {directory}"}
        
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled_pattern = re.compile(pattern, flags)
        
        results = []
        files_searched = 0
        
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            
            # Filter by extension if specified
            if file_extensions and path.suffix not in file_extensions:
                continue
            
            # Skip binary files and very large files
            try:
                if path.stat().st_size > 1_000_000:  # Skip files > 1MB
                    continue
                
                files_searched += 1
                content = path.read_text(encoding="utf-8", errors="ignore")
                
                matches = list(compiled_pattern.finditer(content))
                if matches:
                    # Get context around first match
                    first_match = matches[0]
                    start = max(0, first_match.start() - 100)
                    end = min(len(content), first_match.end() + 100)
                    context = content[start:end].replace("\n", " ")
                    
                    results.append({
                        "path": str(path.relative_to(root)),
                        "match_count": len(matches),
                        "context": context[:200] + "..." if len(context) > 200 else context,
                    })
                    
                    if len(results) >= max_results:
                        break
                        
            except (OSError, UnicodeDecodeError):
                continue
        
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
        "description": "Search for files containing specific content patterns. Useful for finding code files that contain particular functions, classes, or text patterns.",
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
) -> str:
    """Execute the file search tool."""
    result = _search_files_by_content(
        directory=directory,
        pattern=pattern,
        file_extensions=file_extensions,
        max_results=max_results,
        case_sensitive=case_sensitive,
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
