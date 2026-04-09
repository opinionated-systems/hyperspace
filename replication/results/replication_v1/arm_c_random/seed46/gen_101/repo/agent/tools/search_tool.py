"""
Code search tool: search for patterns in the codebase.

Provides grep-like functionality with better output formatting
and cross-file search capabilities.
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
            "Search for patterns in the codebase. "
            "Supports regex patterns and file filtering. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path). Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py'). Optional.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default 50.",
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
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in the codebase."""
    # Validate pattern
    if not pattern or not isinstance(pattern, str):
        return "Error: pattern must be a non-empty string"
    
    # Determine search path
    search_path = path or _ALLOWED_ROOT or os.getcwd()
    search_path = os.path.abspath(search_path)
    
    # Scope check
    if _ALLOWED_ROOT is not None:
        if not search_path.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Path '{search_path}' is outside allowed root '{_ALLOWED_ROOT}'"
    
    # Validate max_results
    if not isinstance(max_results, int) or max_results < 1:
        max_results = 50
    max_results = min(max_results, 200)  # Cap at 200 to prevent huge outputs
    
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n", "-H", "-E"]
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        cmd.extend([pattern, search_path])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )
        
        # Process results
        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        if not lines or lines == ['']:
            return f"No matches found for pattern '{pattern}'"
        
        # Format results
        formatted = []
        count = 0
        
        for line in lines:
            if count >= max_results:
                formatted.append(f"... and {len(lines) - max_results} more matches")
                break
            
            # Parse grep output: file:line:content
            match = re.match(r'^(.+?):(\d+):(.*)$', line)
            if match:
                file_path, line_num, content = match.groups()
                # Make path relative to search root for readability
                try:
                    rel_path = os.path.relpath(file_path, search_path)
                except ValueError:
                    rel_path = file_path
                formatted.append(f"{rel_path}:{line_num}: {content[:100]}")
                count += 1
            elif line.strip():  # Handle lines without match format
                formatted.append(line[:150])
                count += 1
        
        header = f"Found {len(lines)} match(es) for pattern '{pattern}':\n"
        return header + "\n".join(formatted)
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds. Try a more specific pattern or narrower file_pattern."
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:  # grep returns 1 when no matches found
            return f"No matches found for pattern '{pattern}'"
        return f"Error running search: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
