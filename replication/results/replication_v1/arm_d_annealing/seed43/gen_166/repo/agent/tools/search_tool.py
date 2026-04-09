"""
Search tool: find patterns in files using grep/ripgrep.

Provides efficient file searching capabilities for the meta-agent
to locate code patterns, function definitions, and references.
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
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "File extension to filter by (e.g., '.py', '.js'). Optional.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default 50.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default False.",
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
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in (defaults to allowed root)
        file_extension: Filter by file extension (e.g., '.py')
        max_results: Maximum number of results to return
        case_sensitive: Whether search is case sensitive
    
    Returns:
        Search results with file paths, line numbers, and matching lines
    """
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            # Scope check
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        p = Path(search_path)
        if not p.exists():
            return f"Error: {search_path} does not exist."
        
        # Build grep command
        cmd = ["grep", "-rn"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add search path
        cmd.append(search_path)
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 2:
            # grep error
            return f"Error running search: {result.stderr}"
        
        if result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {search_path}"
        
        # Process results
        lines = result.stdout.strip().split("\n")
        if not lines or lines == ['']:
            return f"No matches found for pattern '{pattern}' in {search_path}"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated = True
        else:
            truncated = False
        
        # Format results
        formatted = []
        for line in lines:
            if ':' in line:
                # Format: file:line:content
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    file_path, line_num, content = parts[0], parts[1], parts[2]
                    # Make path relative to search root if possible
                    try:
                        rel_path = os.path.relpath(file_path, search_path)
                    except ValueError:
                        rel_path = file_path
                    formatted.append(f"{rel_path}:{line_num}: {content}")
                else:
                    formatted.append(line)
            else:
                formatted.append(line)
        
        output = "\n".join(formatted)
        if truncated:
            output += f"\n\n... (truncated to {max_results} results, {len(lines)} total matches)"
        
        return f"Found {len(lines)} match(es) for '{pattern}':\n\n{output}"
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or smaller directory."
    except Exception as e:
        return f"Error: {e}"
