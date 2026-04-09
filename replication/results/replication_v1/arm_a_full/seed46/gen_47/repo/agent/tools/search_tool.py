"""
Search tool: search for patterns in files using grep-like functionality.

Provides file search capabilities to help the agent find code patterns,
function definitions, and text across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files. "
            "Supports regex patterns and file filtering. "
            "Returns matching lines with file paths and line numbers. "
            "Can also search for function/class definitions with search_definitions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["pattern", "function", "class"],
                    "description": "Type of search: 'pattern' for regex, 'function' for function defs, 'class' for class defs.",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    max_results: int = 50,
    search_type: str = "pattern",
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The regex pattern to search for (or name for function/class search)
        path: Directory or file to search in (absolute path)
        file_extension: Optional file extension filter (e.g., '.py')
        max_results: Maximum number of results to return
        search_type: Type of search - 'pattern', 'function', or 'class'
    
    Returns:
        String with search results or error message
    """
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        results = []
        count = 0
        
        # Build regex pattern based on search type
        if search_type == "function":
            # Match function definitions: def name(... or async def name(...
            search_pattern = rf"^\s*(?:async\s+)?def\s+{re.escape(pattern)}\s*\("
            search_desc = f"function '{pattern}'"
        elif search_type == "class":
            # Match class definitions: class Name(... or class Name:
            search_pattern = rf"^\s*class\s+{re.escape(pattern)}\s*(?:\(|:)"
            search_desc = f"class '{pattern}'"
        else:
            # Regular pattern search
            search_pattern = pattern
            search_desc = f"pattern '{pattern}'"
        
        # Compile regex pattern
        try:
            regex = re.compile(search_pattern, re.MULTILINE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        
        # Determine files to search
        if p.is_file():
            files = [p]
        else:
            # Find all files recursively
            if file_extension:
                files = list(p.rglob(f"*{file_extension}"))
            else:
                files = [f for f in p.rglob("*") if f.is_file()]
        
        # Search in each file
        for file_path in files:
            # Skip binary files and hidden files
            if file_path.name.startswith("."):
                continue
            if ".pyc" in file_path.suffix or file_path.suffix in [".bin", ".exe", ".so", ".dylib"]:
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        # Format: file:line: content
                        rel_path = file_path.relative_to(p) if p.is_dir() else file_path.name
                        results.append(f"{rel_path}:{line_num}: {line.strip()}")
                        count += 1
                        
                        if count >= max_results:
                            break
                
                if count >= max_results:
                    break
                    
            except (IOError, OSError, UnicodeDecodeError):
                # Skip files that can't be read
                continue
        
        if not results:
            return f"No matches found for {search_desc} in {path}"
        
        # Format output
        header = f"Found {len(results)} match(es) for {search_desc}:\n"
        truncated = "\n[Results truncated]\n" if count >= max_results else ""
        
        return header + "\n".join(results) + truncated
        
    except Exception as e:
        return f"Error during search: {e}"
