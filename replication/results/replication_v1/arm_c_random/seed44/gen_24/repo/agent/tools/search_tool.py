"""
Search tool: search for patterns in files.

Provides grep-like functionality for searching through codebases.
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
            "Search for patterns in files using grep or regex. "
            "Useful for finding code patterns, function definitions, or references. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex or string).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to current directory.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
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


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for pattern in files."""
    try:
        search_path = Path(path)
        
        # Ensure path is absolute
        if not search_path.is_absolute():
            search_path = Path(os.getcwd()) / search_path
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(search_path))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if not search_path.exists():
            return f"Error: path {path} does not exist."
        
        results = []
        
        if search_path.is_file():
            # Search in single file
            files_to_search = [search_path]
        else:
            # Search in directory
            if file_pattern:
                files_to_search = list(search_path.rglob(file_pattern))
            else:
                # Default: search Python files and common text files
                files_to_search = []
                for ext in ["*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.js", "*.ts", "*.html", "*.css"]:
                    files_to_search.extend(search_path.rglob(ext))
        
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for file_path in files_to_search:
            if not file_path.is_file():
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                
                for i, line in enumerate(lines, 1):
                    try:
                        if re.search(pattern, line, flags):
                            # Show context: line number and content
                            rel_path = file_path.relative_to(search_path) if file_path.is_relative_to(search_path) else file_path
                            results.append(f"{rel_path}:{i}: {line.strip()}")
                    except re.error:
                        # If regex fails, try literal string match
                        if pattern in line:
                            rel_path = file_path.relative_to(search_path) if file_path.is_relative_to(search_path) else file_path
                            results.append(f"{rel_path}:{i}: {line.strip()}")
            except (IOError, OSError):
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        # Limit results to prevent overwhelming output
        if len(results) > 100:
            return "\n".join(results[:50]) + f"\n... ({len(results) - 50} more matches) ...\n" + "\n".join(results[-50:])
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error during search: {e}"
