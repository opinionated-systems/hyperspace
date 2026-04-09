"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore and understand codebases.
Supports regex patterns, file filtering, and recursive directory search.
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
            "Search for patterns in files using grep. "
            "Supports regex patterns, file filtering by extension, "
            "and recursive directory search. "
            "Returns matching lines with file paths and line numbers."
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
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                    "default": 50,
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in (absolute path)
        file_extension: Optional file extension filter (e.g., '.py')
        max_results: Maximum number of results to return
    
    Returns:
        Search results with file paths, line numbers, and matching lines
    """
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        # Validate pattern
        if not pattern or not pattern.strip():
            return "Error: Empty search pattern."
        
        # Build grep command
        grep_cmd = ["grep", "-n", "-r", "-E", "--include", file_extension or "*"]
        
        # Add pattern and path
        grep_cmd.extend([pattern, str(p)])
        
        # Run grep
        result = subprocess.run(
            grep_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {p}"
        
        if result.returncode != 0:
            return f"Error searching: {result.stderr}"
        
        # Process results
        lines = result.stdout.strip().split("\n")
        
        if not lines or lines == ['']:
            return f"No matches found for pattern '{pattern}' in {p}"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (truncated, showing {max_results} of {len(lines)}+ matches)"
        else:
            truncated_msg = ""
        
        # Format results
        formatted = []
        for line in lines:
            if ':' in line:
                # Parse file:line:content format
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    file_path, line_num, content = parts[0], parts[1], parts[2]
                    # Make path relative to search root for cleaner output
                    try:
                        rel_path = Path(file_path).relative_to(p if p.is_dir() else p.parent)
                        formatted.append(f"{rel_path}:{line_num}: {content}")
                    except ValueError:
                        formatted.append(f"{file_path}:{line_num}: {content}")
                else:
                    formatted.append(line)
            else:
                formatted.append(line)
        
        output = "\n".join(formatted) + truncated_msg
        return f"Found {len(lines)} match(es) for pattern '{pattern}':\n{output}"
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30s. Try a more specific pattern or narrower path."
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    except Exception as e:
        return f"Error during search: {e}"
