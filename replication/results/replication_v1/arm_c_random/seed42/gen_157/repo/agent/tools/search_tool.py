"""
Search tool: find patterns in files using grep-like functionality.

Provides content search capabilities across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for patterns in files using regex or literal string matching. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The search pattern (regex or literal string)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to search in. Defaults to current directory.",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Whether to treat pattern as regex. Defaults to True.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search is case sensitive. Defaults to True.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Defaults to 50.",
                    },
                },
                "required": ["pattern"],
            },
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    regex: bool = True,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: Search pattern (regex or literal)
        path: Directory or file to search in
        regex: Whether pattern is a regex
        case_sensitive: Whether search is case sensitive
        max_results: Maximum results to return
    
    Returns:
        Formatted search results
    """
    if not os.path.exists(path):
        return f"Error: Path does not exist: {path}"
    
    results = []
    count = 0
    
    # Compile regex pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    if regex:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
    else:
        # Escape literal string for regex use
        compiled = re.compile(re.escape(pattern), flags)
    
    # Walk through directory or search single file
    if os.path.isfile(path):
        files = [path]
    else:
        files = []
        for root, _, filenames in os.walk(path):
            # Skip hidden directories and __pycache__
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if '__pycache__' in root:
                continue
            for filename in filenames:
                # Skip binary files and hidden files
                if filename.startswith('.'):
                    continue
                if any(filename.endswith(ext) for ext in ['.pyc', '.so', '.dll', '.exe']):
                    continue
                files.append(os.path.join(root, filename))
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if compiled.search(line):
                        # Truncate long lines
                        display_line = line.rstrip()
                        if len(display_line) > 200:
                            display_line = display_line[:200] + "..."
                        results.append(f"{filepath}:{line_num}: {display_line}")
                        count += 1
                        if count >= max_results:
                            break
                if count >= max_results:
                    break
        except (IOError, OSError, UnicodeDecodeError):
            continue
    
    if not results:
        return f"No matches found for pattern: {pattern}"
    
    header = f"Found {len(results)} match(es) for pattern: {pattern}\n"
    if count >= max_results:
        header = f"Found {max_results}+ match(es) for pattern: {pattern} (showing first {max_results})\n"
    
    return header + "\n".join(results)
