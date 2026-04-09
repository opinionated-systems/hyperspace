"""
Search tool: search for patterns in files using grep and find.

Provides powerful search capabilities for the agent to locate code,
references, and patterns across the codebase.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep and find. "
            "Supports regex patterns, file type filtering, and case-insensitive search. "
            "Results are limited to prevent overwhelming output."
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
                    "description": "Directory to search in (absolute path). Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', '*.js'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive. Default: false.",
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


def _validate_path(path: str) -> tuple[bool, str]:
    """Validate that path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True, path
    
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, resolved


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for patterns in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory to search in (defaults to allowed root or current dir)
        file_pattern: File pattern to match (e.g., '*.py')
        case_sensitive: Whether search is case-sensitive
        max_results: Maximum number of results to return
    
    Returns:
        JSON string with search results
    """
    result = {
        "pattern": pattern,
        "path": path,
        "file_pattern": file_pattern,
        "case_sensitive": case_sensitive,
        "success": False,
    }
    
    # Validate pattern
    if not isinstance(pattern, str) or not pattern.strip():
        result["error"] = "Invalid pattern: must be a non-empty string"
        return json.dumps(result)
    
    # Determine search path
    search_path = path or _ALLOWED_ROOT or os.getcwd()
    
    # Validate path
    valid, msg = _validate_path(search_path)
    if not valid:
        result["error"] = msg
        return json.dumps(result)
    
    search_path = msg
    
    if not os.path.exists(search_path):
        result["error"] = f"Path does not exist: {search_path}"
        return json.dumps(result)
    
    if not os.path.isdir(search_path):
        result["error"] = f"Path is not a directory: {search_path}"
        return json.dumps(result)
    
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add case sensitivity flag
        if not case_sensitive:
            cmd.append("-i")
        
        # Add file pattern if specified
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Exclude binary files and hidden directories
        cmd.extend(["--binary-files=without-match", "--exclude-dir=.*"])
        
        # Add pattern and path
        cmd.extend([pattern, search_path])
        
        # Run search
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )
        
        # Parse results
        lines = proc.stdout.strip().split("\n") if proc.stdout else []
        
        # Filter out empty lines and limit results
        results = []
        for line in lines:
            if not line:
                continue
            # Parse grep output: path:line_number:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                file_path, line_num, content = parts[0], parts[1], parts[2]
                # Make path relative to search_path if possible
                try:
                    rel_path = os.path.relpath(file_path, search_path)
                except ValueError:
                    rel_path = file_path
                results.append({
                    "file": rel_path,
                    "line": int(line_num) if line_num.isdigit() else 0,
                    "content": content[:200],  # Truncate long lines
                })
            
            if len(results) >= max_results:
                break
        
        result["success"] = True
        result["total_matches"] = len(lines)
        result["results_returned"] = len(results)
        result["truncated"] = len(lines) > max_results
        result["matches"] = results
        
        if proc.returncode == 1 and not results:
            # No matches found (grep returns 1 when no matches)
            result["message"] = "No matches found"
        elif proc.returncode != 0 and proc.stderr:
            result["warning"] = proc.stderr[:500]
        
    except subprocess.TimeoutExpired:
        result["error"] = "Search timed out after 30 seconds"
        result["partial_results"] = results if 'results' in dir() else []
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
    
    return json.dumps(result)
