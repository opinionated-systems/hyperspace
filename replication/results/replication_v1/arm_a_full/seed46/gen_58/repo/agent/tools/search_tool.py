"""
Search tool: search for patterns in files using grep/find.

Provides efficient codebase search capabilities for the meta agent
to locate relevant code sections during self-improvement.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns, file type filtering, and recursive search. "
            "Useful for finding code patterns, function definitions, or references."
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
                    "description": "Directory or file to search in (absolute path). Default: allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', '*.md'). Default: all files.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories. Default: true.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search. Default: false (case-insensitive).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate_output(output: str, max_lines: int = 100, max_chars: int = 10000) -> str:
    """Truncate output to prevent context overflow."""
    if len(output) > max_chars:
        half = max_chars // 2
        output = output[:half] + "\n... [output truncated, too long] ...\n" + output[-half:]
    
    lines = output.split("\n")
    if len(lines) > max_lines:
        output = "\n".join(lines[:max_lines // 2]) + "\n... [lines truncated] ...\n" + "\n".join(lines[-max_lines // 2:])
    
    return output


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for patterns in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in (default: allowed root)
        file_pattern: File pattern to match (e.g., '*.py')
        recursive: Search recursively
        case_sensitive: Case-sensitive search
        max_results: Maximum results to return
        
    Returns:
        Search results with file paths and matching lines
    """
    # Validate and resolve path
    search_path = path or _ALLOWED_ROOT
    if not search_path:
        return "Error: No search path specified and no allowed root set."
    
    search_path = os.path.abspath(search_path)
    
    # Security check
    if _ALLOWED_ROOT is not None:
        if not search_path.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    if not os.path.exists(search_path):
        return f"Error: path does not exist: {search_path}"
    
    # Build grep command
    cmd = ["grep"]
    
    # Add options
    if not case_sensitive:
        cmd.append("-i")  # Case insensitive
    if recursive:
        cmd.append("-r")
    cmd.append("-n")  # Line numbers
    cmd.append("-H")  # Print filename
    cmd.append("--include")  # File pattern
    cmd.append(file_pattern or "*")
    
    # Add pattern and path
    cmd.append(pattern)
    cmd.append(search_path)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Success - found matches
            output = result.stdout
            if not output.strip():
                return f"No matches found for pattern '{pattern}'"
            
            # Count matches
            lines = output.strip().split("\n")
            total_matches = len(lines)
            
            # Truncate if needed
            output = _truncate_output(output, max_lines=max_results)
            
            return f"Found {total_matches} matches for pattern '{pattern}':\n{output}"
        
        elif result.returncode == 1:
            # No matches found (grep returns 1 when no matches)
            return f"No matches found for pattern '{pattern}'"
        
        else:
            # Error
            return f"Search error (exit code {result.returncode}): {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or smaller path."
    except FileNotFoundError:
        # grep not available, try with find + python
        return _fallback_search(pattern, search_path, file_pattern, recursive, case_sensitive, max_results)
    except Exception as e:
        return f"Error during search: {e}"


def _fallback_search(
    pattern: str,
    search_path: str,
    file_pattern: str | None,
    recursive: bool,
    case_sensitive: bool,
    max_results: int,
) -> str:
    """Fallback search using Python when grep is not available."""
    import fnmatch
    import re
    
    results = []
    match_count = 0
    
    # Compile regex
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    # File pattern matching
    file_glob = file_pattern or "*"
    
    def should_search_file(filename: str) -> bool:
        return fnmatch.fnmatch(filename, file_glob)
    
    try:
        if os.path.isfile(search_path):
            # Single file
            files_to_search = [search_path]
        else:
            # Directory
            if recursive:
                files_to_search = []
                for root, _, files in os.walk(search_path):
                    for f in files:
                        if should_search_file(f):
                            files_to_search.append(os.path.join(root, f))
            else:
                files_to_search = [
                    os.path.join(search_path, f)
                    for f in os.listdir(search_path)
                    if os.path.isfile(os.path.join(search_path, f)) and should_search_file(f)
                ]
        
        # Search files
        for filepath in files_to_search:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{filepath}:{line_num}:{line.rstrip()}")
                            match_count += 1
                            if match_count >= max_results * 2:  # Limit raw results
                                break
            except (IOError, OSError):
                continue
            
            if match_count >= max_results * 2:
                break
        
        if not results:
            return f"No matches found for pattern '{pattern}'"
        
        output = "\n".join(results[:max_results])
        if len(results) > max_results:
            output += f"\n... and {len(results) - max_results} more matches"
        
        return f"Found {len(results)} matches for pattern '{pattern}':\n{output}"
    
    except Exception as e:
        return f"Error during fallback search: {e}"
