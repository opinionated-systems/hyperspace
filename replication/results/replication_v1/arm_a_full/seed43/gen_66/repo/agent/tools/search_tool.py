"""
Search tool: search for patterns in files and directories.

Provides grep-like functionality to find text patterns across the codebase.
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
            "Search for patterns in files and directories. "
            "Supports regex patterns and can search recursively. "
            "Returns matching lines with file paths and line numbers. "
            "Can also search for multiple patterns at once using multi_search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["search", "multi_search"],
                    "description": "The search command to run. Default: search.",
                },
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported). Required for 'search' command.",
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search patterns for multi_search command.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory to search in. Defaults to current directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in directories. Default: True.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default: False.",
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


def _search_single_pattern(
    pattern: str,
    path: Path,
    recursive: bool,
    file_extension: str | None,
    case_sensitive: bool,
    max_results: int = 100,
) -> tuple[list[str], str | None]:
    """Search for a single pattern in files.
    
    Returns (results, error_message). If error_message is not None, results is empty.
    """
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return [], f"Error: Invalid regex pattern '{pattern}': {e}"
    
    results = []
    
    if path.is_file():
        files_to_search = [path]
    elif path.is_dir():
        if recursive:
            files_to_search = list(path.rglob("*"))
        else:
            files_to_search = list(path.iterdir())
        files_to_search = [f for f in files_to_search if f.is_file()]
    else:
        return [], f"Error: {path} does not exist."
    
    # Filter by extension if specified
    if file_extension:
        files_to_search = [f for f in files_to_search if f.suffix == file_extension]
    
    # Skip binary files and common non-text files
    skip_extensions = {'.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', 
                     '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
                     '.pdf', '.zip', '.tar', '.gz', '.bz2', '.7z',
                     '.db', '.sqlite', '.sqlite3'}
    
    for file_path in files_to_search:
        if file_path.suffix in skip_extensions:
            continue
        
        try:
            if file_path.stat().st_size > 1_000_000:
                continue
            
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                if compiled_pattern.search(line):
                    if len(line) > 200:
                        line = line[:100] + " ... " + line[-100:]
                    results.append(f"{file_path}:{line_num}: {line}")
                    
                    if len(results) >= max_results:
                        return results, None
                        
        except (IOError, OSError, UnicodeDecodeError):
            continue
    
    return results, None


def tool_function(
    pattern: str,
    path: str | None = None,
    recursive: bool = True,
    file_extension: str | None = None,
    case_sensitive: bool = False,
    command: str = "search",
    patterns: list[str] | None = None,
) -> str:
    """Search for a pattern in files.
    
    Returns matching lines with file paths and line numbers.
    
    Args:
        pattern: The search pattern (regex supported). Required for 'search' command.
        path: Absolute path to file or directory to search in. Defaults to current directory.
        recursive: Whether to search recursively in directories. Default: True.
        file_extension: Optional file extension filter (e.g., '.py', '.txt').
        case_sensitive: Whether the search is case sensitive. Default: False.
        command: The search command to run ('search' or 'multi_search'). Default: 'search'.
        patterns: List of search patterns for multi_search command.
    """
    # Default to current directory if no path provided
    if path is None:
        path = os.getcwd()
    
    p = Path(path)
    
    if not p.is_absolute():
        return f"Error: {path} is not an absolute path."
    
    # Scope check: only allow operations within the allowed root
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    
    max_results = 100  # Limit to prevent overwhelming output
    
    if command == "multi_search":
        # Multi-pattern search
        if not patterns or not isinstance(patterns, list):
            return "Error: 'patterns' list is required for multi_search command."
        
        all_results = {}
        total_matches = 0
        
        for pat in patterns:
            if not pat or not isinstance(pat, str):
                continue
            results, error = _search_single_pattern(
                pat, p, recursive, file_extension, case_sensitive, max_results
            )
            if error:
                return error
            all_results[pat] = results
            total_matches += len(results)
        
        if total_matches == 0:
            return f"No matches found for any of the {len(patterns)} patterns in {p}"
        
        output_parts = [f"Found {total_matches} total matches across {len(patterns)} patterns:"]
        for pat, results in all_results.items():
            if results:
                output_parts.append(f"\n--- Pattern '{pat}' ({len(results)} matches) ---")
                output_parts.extend(results[:50])  # Limit per pattern
                if len(results) >= 50:
                    output_parts.append(f"[Pattern truncated at 50 matches]")
        
        return '\n'.join(output_parts)
    
    else:
        # Single pattern search (default)
        if not pattern:
            return "Error: pattern is required."
        
        results, error = _search_single_pattern(
            pattern, p, recursive, file_extension, case_sensitive, max_results
        )
        
        if error:
            return error
        
        if not results:
            return f"No matches found for pattern '{pattern}' in {p}"
        
        result_text = '\n'.join(results)
        if len(results) >= max_results:
            result_text += f"\n\n[Search truncated at {max_results} matches]"
        
        return f"Found {len(results)} matches for pattern '{pattern}':\n{result_text}"
