"""
Search tool: search for patterns in files using grep/find.

Provides structured file searching with pattern matching,
file type filtering, and result limiting.
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
            "Supports regex patterns, file type filtering, and directory scoping. "
            "Results are limited to prevent context overflow."
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
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
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
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in (defaults to allowed root)
        file_extension: Optional file extension filter (e.g., '.py')
        max_results: Maximum number of results to return
    
    Returns:
        Search results with file paths and matching lines
    """
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No search path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            # Scope check
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if not os.path.exists(search_path):
            return f"Error: Path does not exist: {search_path}"
        
        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include=*"]
        
        # Add file extension filter if specified
        if file_extension:
            ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
            cmd[-1] = f"--include=*{ext}"
        
        # Add pattern and path
        cmd.extend([pattern, search_path])
        
        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns exit code 1 when no matches found
        if result.returncode not in (0, 1):
            return f"Error: grep failed with exit code {result.returncode}: {result.stderr}"
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        if not lines or lines == ['']:
            return f"No matches found for pattern '{pattern}' in {search_path}"
        
        # Limit results
        total_matches = len(lines)
        if total_matches > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total_matches - max_results} more results truncated) ..."
        else:
            truncated_msg = ""
        
        # Format results
        formatted = []
        for line in lines:
            if line:
                # Parse grep output: path:line_num:content
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path, line_num, content = parts[0], parts[1], parts[2]
                    # Make path relative to search path for cleaner output
                    try:
                        rel_path = os.path.relpath(file_path, search_path)
                    except ValueError:
                        rel_path = file_path
                    formatted.append(f"{rel_path}:{line_num}: {content}")
                else:
                    formatted.append(line)
        
        result_text = "\n".join(formatted) + truncated_msg
        return f"Found {total_matches} match(es) for '{pattern}':\n{result_text}"
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30s. Try a more specific pattern or narrower scope."
    except Exception as e:
        return f"Error: {e}"
