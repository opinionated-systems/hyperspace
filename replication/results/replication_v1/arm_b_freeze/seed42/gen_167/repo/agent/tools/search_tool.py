"""
Search tool: grep-like file search functionality.

Provides pattern matching across files in the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any


def _search_with_grep(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Search for pattern using grep command."""
    try:
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        cmd.extend(["--include", f"*.{file_extension}"] if file_extension else [])
        cmd.extend([pattern, path])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode not in (0, 1):  # 0 = matches, 1 = no matches
            return {"error": f"grep failed: {result.stderr}"}
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        lines = [l for l in lines if l][:max_results]
        
        matches = []
        for line in lines:
            # Parse grep output: path:line:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({
                    "file": parts[0],
                    "line": int(parts[1]) if parts[1].isdigit() else 0,
                    "content": parts[2][:200],  # Limit content length
                })
        
        return {
            "matches": matches,
            "total_found": len(lines),
            "returned": len(matches),
            "truncated": len(lines) > max_results,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out (30s limit)"}
    except Exception as e:
        return {"error": f"Search failed: {e}"}


def _search_python(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Fallback Python implementation for searching."""
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_pattern = re.compile(pattern, flags)
    
    matches = []
    total_checked = 0
    
    try:
        base_path = Path(path)
        if not base_path.exists():
            return {"error": f"Path not found: {path}"}
        
        if base_path.is_file():
            files = [base_path]
        else:
            glob_pattern = f"**/*.{file_extension}" if file_extension else "**/*"
            files = list(base_path.glob(glob_pattern))
            files = [f for f in files if f.is_file() and not any(
                part.startswith(".") or part == "__pycache__" 
                for part in f.parts
            )]
        
        for file_path in files:
            if len(matches) >= max_results:
                break
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        total_checked += 1
                        if compiled_pattern.search(line):
                            matches.append({
                                "file": str(file_path),
                                "line": line_num,
                                "content": line.strip()[:200],
                            })
                            if len(matches) >= max_results:
                                break
            except Exception:
                continue
    except Exception as e:
        return {"error": f"Python search failed: {e}"}
    
    return {
        "matches": matches,
        "total_checked": total_checked,
        "returned": len(matches),
        "truncated": len(matches) >= max_results,
    }


def search_files(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = False,
    use_grep: bool = True,
) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional file extension filter (e.g., "py")
        max_results: Maximum number of results to return
        case_sensitive: Whether search is case sensitive
        use_grep: Try using grep command first (faster)
    
    Returns:
        JSON string with search results
    """
    import json
    
    # Validate inputs
    if not pattern:
        return json.dumps({"error": "Pattern cannot be empty"})
    
    if max_results < 1 or max_results > 100:
        max_results = 50
    
    # Try grep first if available
    if use_grep:
        result = _search_with_grep(pattern, path, file_extension, max_results, case_sensitive)
        if "error" not in result:
            return json.dumps(result, indent=2)
    
    # Fallback to Python implementation
    result = _search_python(pattern, path, file_extension, max_results, case_sensitive)
    return json.dumps(result, indent=2)


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "search_files",
        "description": "Search for text patterns in files using regex. Returns file paths, line numbers, and matching content. Useful for finding code patterns, function definitions, or references across the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (e.g., 'def foo', 'class.*Agent', 'TODO')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in (default: current directory)",
                    "default": ".",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension to filter by (e.g., 'py', 'txt', 'md')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-100, default: 50)",
                    "default": 50,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: false)",
                    "default": False,
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(**kwargs) -> str:
    """Execute the search tool."""
    return search_files(**kwargs)
