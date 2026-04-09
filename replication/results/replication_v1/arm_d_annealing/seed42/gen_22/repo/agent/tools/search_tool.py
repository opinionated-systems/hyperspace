"""
Search tool: find files and content within the codebase.

Provides capabilities to search for files by name and content within files.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict:
    """Return the tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files by name or content within a directory. Can find files matching a pattern or containing specific text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["find_files", "search_content"],
                        "description": "The search command to execute. 'find_files' searches for files by name pattern, 'search_content' searches for text within files.",
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory path to search in. Defaults to current directory if not specified.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "For find_files: glob pattern to match (e.g., '*.py', 'test_*.py'). For search_content: regex pattern to search for within files.",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Optional file extension filter for search_content (e.g., '.py', '.txt'). If not specified, searches all files.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Defaults to 50.",
                    },
                },
                "required": ["command", "path", "pattern"],
            },
        },
    }


def _find_files(path: str, pattern: str, max_results: int = 50) -> dict:
    """Find files matching a glob pattern."""
    import fnmatch
    
    if not os.path.exists(path):
        return {
            "success": False,
            "error": f"Path not found: {path}",
            "results": [],
        }
    
    results = []
    count = 0
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                full_path = os.path.join(root, filename)
                results.append(full_path)
                count += 1
                if count >= max_results:
                    break
        if count >= max_results:
            break
    
    return {
        "success": True,
        "command": f"find_files(path='{path}', pattern='{pattern}')",
        "results": results,
        "count": len(results),
        "truncated": count >= max_results,
    }


def _search_content(path: str, pattern: str, file_extension: str | None = None, max_results: int = 50) -> dict:
    """Search for content matching a regex pattern within files."""
    if not os.path.exists(path):
        return {
            "success": False,
            "error": f"Path not found: {path}",
            "results": [],
        }
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "results": [],
        }
    
    results = []
    count = 0
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            # Check file extension filter
            if file_extension and not filename.endswith(file_extension):
                continue
            
            full_path = os.path.join(root, filename)
            
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                matches = []
                for match in regex.finditer(content):
                    # Get context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end].replace('\n', ' ')
                    matches.append({
                        "line": content[:match.start()].count('\n') + 1,
                        "context": context,
                        "match": match.group(),
                    })
                
                if matches:
                    results.append({
                        "file": full_path,
                        "matches": matches,
                    })
                    count += 1
                    if count >= max_results:
                        break
            except (IOError, OSError, UnicodeDecodeError):
                # Skip files that can't be read
                continue
        
        if count >= max_results:
            break
    
    return {
        "success": True,
        "command": f"search_content(path='{path}', pattern='{pattern}', file_extension={file_extension!r})",
        "results": results,
        "count": len(results),
        "truncated": count >= max_results,
    }


def tool_function(**kwargs: Any) -> dict:
    """Execute the search tool."""
    command = kwargs.get("command")
    path = kwargs.get("path", ".")
    pattern = kwargs.get("pattern")
    file_extension = kwargs.get("file_extension")
    max_results = kwargs.get("max_results", 50)
    
    if command == "find_files":
        return _find_files(path, pattern, max_results)
    elif command == "search_content":
        return _search_content(path, pattern, file_extension, max_results)
    else:
        return {
            "success": False,
            "error": f"Unknown command: {command}",
        }
