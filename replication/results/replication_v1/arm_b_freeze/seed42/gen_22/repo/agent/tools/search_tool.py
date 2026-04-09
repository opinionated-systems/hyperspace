"""
Search tool: grep-like functionality for finding patterns in files.

Provides file search capabilities to help the agent locate code patterns,
function definitions, and references across the codebase.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": "Search for patterns in files using regex. Returns matching lines with file paths and line numbers. Useful for finding function definitions, variable references, or any text pattern across the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in. Default is current directory.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "File extension to filter by (e.g., '.py', '.js'). Default searches all files.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return. Default is 50.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for regex pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file path to search in
        file_extension: Optional file extension filter (e.g., '.py')
        max_results: Maximum number of matches to return
    
    Returns:
        Formatted string with matches (file:line:content)
    """
    try:
        search_path = Path(path).expanduser().resolve()
        
        if not search_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        # Compile regex pattern
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        
        matches = []
        files_searched = 0
        
        # Determine files to search
        if search_path.is_file():
            files_to_search = [search_path]
        else:
            # Find all files recursively
            files_to_search = []
            for root, dirs, files in os.walk(search_path):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for filename in files:
                    if filename.startswith('.'):
                        continue
                    if file_extension and not filename.endswith(file_extension):
                        continue
                    files_to_search.append(Path(root) / filename)
        
        # Search each file
        for file_path in files_to_search:
            if len(matches) >= max_results:
                break
                
            try:
                # Skip binary files
                if _is_binary(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    files_searched += 1
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Format: file_path:line_num:content
                            content = line.rstrip()[:100]  # Limit line length
                            rel_path = file_path.relative_to(search_path) if search_path.is_dir() else file_path.name
                            matches.append(f"{rel_path}:{line_num}:{content}")
                            
                            if len(matches) >= max_results:
                                break
            except (IOError, OSError, PermissionError):
                continue
        
        # Format results
        if not matches:
            ext_info = f" (.{file_extension} files)" if file_extension else ""
            return f"No matches found for pattern '{pattern}' in {files_searched} files searched{ext_info}"
        
        truncated = " (truncated)" if len(matches) >= max_results else ""
        header = f"Found {len(matches)} matches for '{pattern}' in {files_searched} files{truncated}:\n"
        return header + "\n".join(matches)
        
    except Exception as e:
        return f"Error during search: {e}"


def _is_binary(file_path: Path, sample_size: int = 8192) -> bool:
    """Check if a file is binary by looking for null bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(sample_size)
            return b'\x00' in chunk
    except:
        return True
