"""
Search tool: search for patterns in files using grep.

Provides file search capabilities to help agents explore and understand codebases.
Supports pattern matching, file filtering, and directory scoping.
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
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path). Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py', '*.js'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default: false.",
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


def _truncate_output(output: str, max_len: int = 10000) -> str:
    """Truncate output if it exceeds max length."""
    if len(output) > max_len:
        lines = output.split("\n")
        # Keep first half and last half of lines
        half_count = len(lines) // 2
        return "\n".join(lines[:half_count]) + f"\n... [{len(lines) - half_count} more lines truncated] ...\n"
    return output


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search for the given pattern.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in (defaults to allowed root)
        file_pattern: Glob pattern to filter files (e.g., '*.py')
        case_sensitive: Whether search is case sensitive
        max_results: Maximum number of results to return
    
    Returns:
        Search results with file paths, line numbers, and matching lines
    """
    # Validate pattern
    if not pattern or not pattern.strip():
        return "Error: Empty search pattern provided."
    
    # Determine search path
    search_path = path
    if search_path is None:
        if _ALLOWED_ROOT is None:
            return "Error: No search path provided and no allowed root set."
        search_path = _ALLOWED_ROOT
    
    # Validate path is absolute
    p = Path(search_path)
    if not p.is_absolute():
        return f"Error: {search_path} is not an absolute path."
    
    # Scope check
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    # Check path exists
    if not p.exists():
        return f"Error: {search_path} does not exist."
    
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add case sensitivity flag
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add search path
        cmd.append(str(p))
        
        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Process results
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        
        # Filter by file pattern if specified
        if file_pattern and lines:
            import fnmatch
            filtered_lines = []
            for line in lines:
                if ":" in line:
                    file_path = line.split(":", 1)[0]
                    if fnmatch.fnmatch(file_path, file_pattern) or fnmatch.fnmatch(os.path.basename(file_path), file_pattern):
                        filtered_lines.append(line)
            lines = filtered_lines
        
        # Limit results
        total_matches = len(lines)
        if total_matches > max_results:
            lines = lines[:max_results]
        
        if not lines:
            return f"No matches found for pattern '{pattern}' in {search_path}"
        
        # Format output
        output = "\n".join(lines)
        output = _truncate_output(output)
        
        result_msg = f"Found {total_matches} match(es) for pattern '{pattern}':\n{output}"
        if total_matches > max_results:
            result_msg += f"\n... ({total_matches - max_results} more results not shown)"
        
        return result_msg
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            # grep returns 1 when no matches found
            return f"No matches found for pattern '{pattern}' in {search_path}"
        return f"Error: Search failed with exit code {e.returncode}: {e.stderr}"
    except Exception as e:
        return f"Error: Search failed: {e}"
