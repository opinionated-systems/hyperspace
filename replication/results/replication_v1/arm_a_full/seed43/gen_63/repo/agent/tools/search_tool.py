"""
Search tool: find files by name or content.

Provides grep-like functionality to search within files,
and find-like functionality to locate files by name pattern.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files by name or content within the codebase. "
            "Use file_pattern to search by filename (e.g., '*.py', 'test_*.py'). "
            "Use content_pattern to search within file contents (grep-like). "
            "Results are limited to prevent context overflow."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to match filenames (e.g., '*.py', 'config*.json').",
                },
                "content_pattern": {
                    "type": "string",
                    "description": "Pattern to search within file contents (supports regex).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": [],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    file_pattern: str | None = None,
    content_pattern: str | None = None,
    path: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command.
    
    At least one of file_pattern or content_pattern must be provided.
    """
    if file_pattern is None and content_pattern is None:
        return "Error: At least one of file_pattern or content_pattern must be provided."
    
    search_path = path or _ALLOWED_ROOT or os.getcwd()
    search_path = os.path.abspath(search_path)
    
    # Scope check
    if _ALLOWED_ROOT is not None:
        if not search_path.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    results = []
    
    try:
        if file_pattern and not content_pattern:
            # File name search only
            results = _search_by_name(search_path, file_pattern, max_results)
        elif content_pattern and not file_pattern:
            # Content search only
            results = _search_by_content(search_path, content_pattern, max_results)
        else:
            # Both: search by name first, then filter by content
            files = _search_by_name(search_path, file_pattern, max_results * 2)
            results = _filter_by_content(files, content_pattern, max_results)
        
        if not results:
            return "No results found."
        
        # Format results
        output_lines = [f"Search results ({len(results)} found):"]
        for result in results[:max_results]:
            output_lines.append(result)
        
        if len(results) > max_results:
            output_lines.append(f"\n... and {len(results) - max_results} more results (truncated)")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"Error during search: {e}"


def _search_by_name(search_path: str, pattern: str, max_results: int) -> list[str]:
    """Find files matching the given pattern."""
    results = []
    
    # Use find command for efficiency
    try:
        cmd = ["find", search_path, "-type", "f", "-name", pattern, "-not", "-path", "*/\.*"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            files = result.stdout.strip().split("\n")
            files = [f for f in files if f]  # Remove empty strings
            
            # Make paths relative to search_path for readability
            for f in files[:max_results]:
                rel_path = os.path.relpath(f, search_path)
                results.append(rel_path)
        
        return results
    except subprocess.TimeoutExpired:
        return ["Error: File search timed out"]
    except Exception as e:
        return [f"Error: {e}"]


def _search_by_content(search_path: str, pattern: str, max_results: int) -> list[str]:
    """Find files containing the given pattern."""
    results = []
    
    try:
        # Use grep for content search
        cmd = [
            "grep", "-r", "-l", "-n",
            "--include=*.*",  # Skip binary files
            pattern,
            search_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode in (0, 1):  # 0 = matches found, 1 = no matches
            files = result.stdout.strip().split("\n")
            files = [f for f in files if f]
            
            for f in files[:max_results]:
                rel_path = os.path.relpath(f, search_path)
                results.append(rel_path)
        
        return results
    except subprocess.TimeoutExpired:
        return ["Error: Content search timed out"]
    except Exception as e:
        return [f"Error: {e}"]


def _filter_by_content(files: list[str], pattern: str, max_results: int) -> list[str]:
    """Filter a list of files by content pattern."""
    results = []
    
    for filepath in files:
        try:
            # Quick check if file contains pattern
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if pattern in content:
                    results.append(os.path.basename(filepath))
                    if len(results) >= max_results:
                        break
        except Exception:
            continue
    
    return results
