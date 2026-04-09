"""
File tool: additional file operations for searching and analyzing code.

Provides utilities for:
- Searching file contents with patterns
- Analyzing code structure
- Finding files by name patterns
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for the file tool."""
    return {
        "name": "file",
        "description": "Search and analyze files in the codebase. Supports searching file contents with regex patterns, finding files by name patterns, and analyzing code structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["search_content", "find_files", "analyze_structure"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory path to search in",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for search_content, glob for find_files)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20)",
                },
            },
            "required": ["command", "path"],
        },
    }


def _search_content(path: str, pattern: str, file_extension: str | None = None, max_results: int = 20) -> str:
    """Search file contents using regex pattern."""
    results = []
    regex = re.compile(pattern)
    count = 0
    
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        for root, _, files in os.walk(base_path):
            # Skip hidden directories and __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in Path(root).parts):
                continue
                
            for filename in files:
                if file_extension and not filename.endswith(file_extension):
                    continue
                    
                filepath = Path(root) / filename
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = list(regex.finditer(content))
                        if matches:
                            for match in matches[:3]:  # Limit matches per file
                                line_num = content[:match.start()].count('\n') + 1
                                line_start = content.rfind('\n', 0, match.start()) + 1
                                line_end = content.find('\n', match.start())
                                if line_end == -1:
                                    line_end = len(content)
                                line_content = content[line_start:line_end].strip()
                                results.append(f"{filepath}:{line_num}: {line_content}")
                                count += 1
                                if count >= max_results:
                                    break
                        if count >= max_results:
                            break
                except (IOError, OSError):
                    continue
            if count >= max_results:
                break
        
        if not results:
            return f"No matches found for pattern '{pattern}'"
        return '\n'.join(results[:max_results])
    except Exception as e:
        return f"Error searching content: {e}"


def _find_files(path: str, pattern: str, max_results: int = 20) -> str:
    """Find files by name pattern (glob)."""
    results = []
    count = 0
    
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        for root, _, files in os.walk(base_path):
            # Skip hidden directories and __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in Path(root).parts):
                continue
                
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    results.append(str(Path(root) / filename))
                    count += 1
                    if count >= max_results:
                        break
            if count >= max_results:
                break
        
        if not results:
            return f"No files found matching pattern '{pattern}'"
        return '\n'.join(results[:max_results])
    except Exception as e:
        return f"Error finding files: {e}"


def _analyze_structure(path: str, max_results: int = 20) -> str:
    """Analyze code structure - list Python classes and functions."""
    results = []
    count = 0
    
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        for root, _, files in os.walk(base_path):
            # Skip hidden directories and __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in Path(root).parts):
                continue
                
            for filename in files:
                if not filename.endswith('.py'):
                    continue
                    
                filepath = Path(root) / filename
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Find classes
                    class_matches = re.finditer(r'^class\s+(\w+)', content, re.MULTILINE)
                    for match in class_matches:
                        line_num = content[:match.start()].count('\n') + 1
                        results.append(f"{filepath}:{line_num}: class {match.group(1)}")
                        count += 1
                        if count >= max_results:
                            break
                    
                    # Find functions
                    if count < max_results:
                        func_matches = re.finditer(r'^def\s+(\w+)', content, re.MULTILINE)
                        for match in func_matches:
                            line_num = content[:match.start()].count('\n') + 1
                            results.append(f"{filepath}:{line_num}: def {match.group(1)}")
                            count += 1
                            if count >= max_results:
                                break
                                
                    if count >= max_results:
                        break
                except (IOError, OSError):
                    continue
            if count >= max_results:
                break
        
        if not results:
            return f"No Python code structure found in '{path}'"
        return '\n'.join(results[:max_results])
    except Exception as e:
        return f"Error analyzing structure: {e}"


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
    file_extension: str | None = None,
    max_results: int = 20,
) -> str:
    """Execute the file tool."""
    if command == "search_content":
        if pattern is None:
            return "Error: 'pattern' is required for search_content command"
        return _search_content(path, pattern, file_extension, max_results)
    elif command == "find_files":
        if pattern is None:
            return "Error: 'pattern' is required for find_files command"
        return _find_files(path, pattern, max_results)
    elif command == "analyze_structure":
        return _analyze_structure(path, max_results)
    else:
        return f"Error: Unknown command '{command}'"
