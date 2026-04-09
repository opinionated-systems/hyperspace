"""
Search tool: search for files and content within the codebase.

Provides grep-like functionality to find files and content patterns.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator


def _search_files(
    pattern: str,
    path: str,
    file_pattern: str = "*",
    max_results: int = 50,
) -> Iterator[dict]:
    """Search for pattern in files under path.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory to search in
        file_pattern: Glob pattern for files to search (e.g., "*.py")
        max_results: Maximum number of matches to return
    
    Yields:
        Dict with file, line, and match info
    """
    search_path = Path(path)
    if not search_path.exists():
        return
    
    count = 0
    try:
        compiled_pattern = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        yield {"error": f"Invalid regex pattern: {e}"}
        return
    
    for file_path in search_path.rglob(file_pattern):
        if not file_path.is_file():
            continue
        
        # Skip binary files and large files
        try:
            if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                continue
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except (IOError, UnicodeDecodeError):
            continue
        
        for match in compiled_pattern.finditer(content):
            if count >= max_results:
                return
            
            # Find line number
            line_num = content[:match.start()].count('\n') + 1
            # Get context (line content)
            lines = content.split('\n')
            line_content = lines[line_num - 1] if line_num <= len(lines) else ""
            
            yield {
                "file": str(file_path),
                "line": line_num,
                "match": match.group(0),
                "context": line_content.strip()[:100],  # Limit context length
            }
            count += 1


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "search",
        "description": "Search for files and content patterns in the codebase. Uses regex pattern matching.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to search (e.g., '*.py', '*.js'). Default is all files.",
                    "default": "*",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default 50)",
                    "default": 50,
                },
            },
            "required": ["pattern", "path"],
        },
    }


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str = "*",
    max_results: int = 50,
) -> str:
    """Execute search and return formatted results."""
    results = list(_search_files(pattern, path, file_pattern, max_results))
    
    if not results:
        return "No matches found."
    
    if len(results) == 1 and "error" in results[0]:
        return f"Error: {results[0]['error']}"
    
    lines = [f"Found {len(results)} match(es):"]
    for r in results:
        lines.append(f"  {r['file']}:{r['line']}: {r['context']}")
    
    if len(results) >= max_results:
        lines.append(f"(Results truncated to {max_results} matches)")
    
    return "\n".join(lines)
