"""
File search tool: search for files by name or content.

Provides functionality to search within the repository for files
matching specific patterns or containing specific text.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_search",
        "description": "Search for files by name pattern or content within a directory. Useful for finding files when you don't know their exact location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search in",
                },
                "name_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to match file names (e.g., '*.py', 'test_*.py')",
                },
                "content_pattern": {
                    "type": "string",
                    "description": "Optional regex pattern to search for in file contents",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["path"],
        },
    }


def tool_function(
    path: str,
    name_pattern: str | None = None,
    content_pattern: str | None = None,
    max_results: int = 20,
) -> str:
    """Search for files by name pattern or content.
    
    Args:
        path: Directory path to search in
        name_pattern: Optional glob pattern for file names
        content_pattern: Optional regex pattern for file contents
        max_results: Maximum number of results to return
    
    Returns:
        String with search results
    """
    search_path = Path(path).expanduser().resolve()
    
    if not search_path.exists():
        return f"Error: Path '{path}' does not exist"
    
    if not search_path.is_dir():
        return f"Error: Path '{path}' is not a directory"
    
    results = []
    count = 0
    
    try:
        # Walk through directory
        for root, dirs, files in os.walk(search_path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for filename in files:
                if filename.startswith('.'):
                    continue
                
                file_path = Path(root) / filename
                
                # Check name pattern
                if name_pattern and not file_path.match(name_pattern):
                    continue
                
                # Check content pattern
                if content_pattern:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if not re.search(content_pattern, content):
                                continue
                    except (IOError, OSError):
                        continue
                
                # Add to results
                rel_path = file_path.relative_to(search_path)
                results.append(str(rel_path))
                count += 1
                
                if count >= max_results:
                    break
            
            if count >= max_results:
                break
        
        if not results:
            return f"No files found matching criteria in '{path}'"
        
        result_str = f"Found {len(results)} file(s) in '{path}':\n"
        if count >= max_results:
            result_str = f"Found {len(results)}+ file(s) in '{path}' (showing first {max_results}):\n"
        
        result_str += "\n".join(f"  - {r}" for r in results)
        return result_str
        
    except Exception as e:
        return f"Error searching files: {e}"
