"""
Search tool for finding files and content within the codebase.

Provides grep-like and find-like functionality to help the agent
locate relevant code quickly.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for the search tool."""
    return {
        "name": "search",
        "description": "Search for files and content within the codebase. Supports grep (content search) and find (file search) operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["grep", "find"],
                    "description": "Type of search: 'grep' for content search, 'find' for file search",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for grep, glob pattern for find)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "For grep: only search files matching this pattern (e.g., '*.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                },
            },
            "required": ["operation", "pattern"],
        },
    }


def tool_function(
    operation: str,
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute search operation.
    
    Args:
        operation: 'grep' for content search, 'find' for file search
        pattern: Search pattern
        path: Directory or file to search in
        file_pattern: For grep, filter by file pattern
        max_results: Maximum results to return
    
    Returns:
        Search results as formatted string
    """
    if operation == "grep":
        return _grep_search(pattern, path, file_pattern, max_results)
    elif operation == "find":
        return _find_files(pattern, path, max_results)
    else:
        return f"Error: Unknown operation '{operation}'. Use 'grep' or 'find'."


def _grep_search(pattern: str, path: str, file_pattern: str | None, max_results: int) -> str:
    """Search file contents using grep."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    
    cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", "-E", pattern, path]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        lines = [l for l in lines if l]
        
        if not lines:
            return f"No matches found for pattern '{pattern}'"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated = True
        else:
            truncated = False
        
        output = "\n".join(lines)
        if truncated:
            output += f"\n... (truncated, showing {max_results} of {len(lines)}+ matches)"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error during search: {e}"


def _find_files(pattern: str, path: str, max_results: int) -> str:
    """Find files by name pattern."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    
    matched_files = []
    
    # Convert glob pattern to regex
    regex_pattern = pattern.replace(".", r"\.")
    regex_pattern = regex_pattern.replace("*", ".*")
    regex_pattern = regex_pattern.replace("?", ".")
    regex = re.compile(regex_pattern)
    
    try:
        for root, dirs, files in os.walk(path):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]
            
            for filename in files:
                if regex.search(filename):
                    full_path = os.path.join(root, filename)
                    matched_files.append(full_path)
                    
                    if len(matched_files) >= max_results:
                        break
            
            if len(matched_files) >= max_results:
                break
        
        if not matched_files:
            return f"No files found matching pattern '{pattern}'"
        
        output = "\n".join(matched_files)
        if len(matched_files) >= max_results:
            output += f"\n... (showing first {max_results} matches)"
        
        return output
        
    except Exception as e:
        return f"Error during file search: {e}"
