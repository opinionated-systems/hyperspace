"""
Search tool for finding patterns in files.

Provides grep-like functionality to search for text patterns across files.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for a regex pattern in files. "
            "Returns file paths, line numbers, and matching content. "
            "Limited to 50 results. "
            "Useful for finding function definitions, imports, or any text pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (e.g., 'def foo', 'class.*Agent')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in. Defaults to current directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories. Defaults to True.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(pattern: str, path: str = ".", recursive: bool = True) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        recursive: Whether to search recursively in subdirectories

    Returns:
        Formatted string with search results
    """
    results = []
    
    if not os.path.exists(path):
        return f"Error: Path does not exist: {path}"
    
    try:
        if recursive and os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for filename in files:
                    if filename.startswith('.') or filename.endswith('.pyc'):
                        continue
                    
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        for i, line in enumerate(content.split('\n'), 1):
                            if re.search(pattern, line):
                                results.append({
                                    "file": filepath,
                                    "line": i,
                                    "content": line[:200],  # Limit line length
                                })
                    except Exception:
                        continue
        else:
            # Single file search
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for i, line in enumerate(content.split('\n'), 1):
                if re.search(pattern, line):
                    results.append({
                        "file": path,
                        "line": i,
                        "content": line[:200],
                    })
        
        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        # Format results
        output = [f"Found {len(results)} matches for pattern '{pattern}':\n"]
        for match in results[:50]:  # Limit to 50 results
            output.append(f"{match['file']}:{match['line']}: {match['content']}")
        
        if len(results) > 50:
            output.append(f"\n... and {len(results) - 50} more matches")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error: {e}"
