"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns within files,
with support for regex and file filtering.
"""

from __future__ import annotations

import os
import re
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata matching the expected schema."""
    return {
        "name": "search",
        "description": "Search for patterns in files within a directory. Returns matching file paths with line numbers and context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: current directory)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py', '*.md')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 50)",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory to search in
        file_pattern: Optional glob pattern to filter files
        max_results: Maximum number of matches to return
        
    Returns:
        String with search results
    """
    import fnmatch
    
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a valid directory"
    
    matches = []
    count = 0
    
    try:
        compiled_pattern = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
        
        for filename in files:
            if filename.startswith('.'):
                continue
                
            # Apply file pattern filter if specified
            if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                continue
                
            filepath = os.path.join(root, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for match in compiled_pattern.finditer(content):
                    if count >= max_results:
                        break
                        
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Get context (line content)
                    lines = content.split('\n')
                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                    
                    # Truncate long lines
                    if len(line_content) > 100:
                        line_content = line_content[:97] + "..."
                    
                    matches.append(f"{filepath}:{line_num}: {line_content}")
                    count += 1
                    
                if count >= max_results:
                    break
                    
            except (IOError, OSError, UnicodeDecodeError):
                continue
                
        if count >= max_results:
            break
    
    if not matches:
        return f"No matches found for pattern '{pattern}'"
    
    result = f"Found {len(matches)} match(es) for pattern '{pattern}':\n"
    result += "\n".join(matches)
    
    if count >= max_results:
        result += f"\n\n(Results truncated to {max_results} matches)"
    
    return result
