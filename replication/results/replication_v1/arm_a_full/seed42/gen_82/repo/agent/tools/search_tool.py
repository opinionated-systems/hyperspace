"""
Search tool: find files by name or content within the repository.

Provides grep-like functionality to search for patterns in file contents
and find files by name patterns.
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
            "Search for files by name pattern or content within the repository. "
            "Useful for finding where specific functions, classes, or patterns are defined."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex for content, glob for filename).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: repo root).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Whether to search in file contents or filenames.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["pattern", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for searches."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for files by name or content.
    
    Args:
        pattern: Search pattern (regex for content, glob for filename)
        search_type: "content" or "filename"
        path: Directory to search in (default: allowed root)
        file_extension: Optional extension filter (e.g., ".py")
        max_results: Maximum results to return
    
    Returns:
        Formatted search results
    """
    search_path = path or _ALLOWED_ROOT or "."
    
    if not _is_within_allowed(search_path):
        return f"Error: Search path '{search_path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist."
    
    results: list[str] = []
    
    if search_type == "filename":
        # Use find command for filename search
        cmd = ["find", search_path, "-type", "f", "-name", pattern]
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            
            if file_extension:
                files = [f for f in files if f.endswith(file_extension)]
            
            results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out."
        except Exception as e:
            return f"Error during search: {e}"
    
    elif search_type == "content":
        # Use grep for content search
        # Build grep command with file extension filter if provided
        include_pattern = f"*{file_extension}" if file_extension else "*"
        
        cmd = [
            "grep", "-r", "-n", "-l", "--include", include_pattern,
            pattern, search_path
        ]
        
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out."
        except Exception as e:
            return f"Error during search: {e}"
    
    else:
        return f"Error: Invalid search_type '{search_type}'. Use 'content' or 'filename'."
    
    if not results:
        return f"No results found for pattern '{pattern}' (type: {search_type})."
    
    # Format results
    output_lines = [
        f"Found {len(results)} result(s) for pattern '{pattern}' (type: {search_type}):",
        "",
    ]
    
    for i, result in enumerate(results, 1):
        # Show relative path if within allowed root
        if _ALLOWED_ROOT and result.startswith(_ALLOWED_ROOT):
            rel_path = os.path.relpath(result, _ALLOWED_ROOT)
            output_lines.append(f"{i}. {rel_path}")
        else:
            output_lines.append(f"{i}. {result}")
    
    if len(results) == max_results:
        output_lines.append("")
        output_lines.append(f"(Results limited to {max_results}. Refine your search for more specific results.)")
    
    return "\n".join(output_lines)
