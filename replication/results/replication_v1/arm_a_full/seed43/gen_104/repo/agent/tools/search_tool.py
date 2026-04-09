"""
Search tool: search for patterns in files using grep.

Provides a convenient way to search file contents without
needing to construct complex bash grep commands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns, file filtering, and recursive search. "
            "Returns matching lines with line numbers. "
            "Can show context lines before and after matches."
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
                    "description": "Directory or file to search in. Defaults to current directory.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to filter files (e.g., '*.py').",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories. Default: true.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search. Default: true.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before and after each match (like grep -C). Default: 0.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = True,
    max_results: int = 50,
    context_lines: int = 0,
) -> str:
    """Search for a pattern in files."""
    if not pattern:
        return "Error: pattern is required"

    p = Path(path)
    if not p.exists():
        return f"Error: path '{path}' does not exist"

    # Build grep command
    cmd = ["grep"]
    
    # Add options
    if recursive:
        cmd.append("-r")
    if not case_sensitive:
        cmd.append("-i")
    # Always show line numbers
    cmd.append("-n")
    # Add context lines if specified
    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
    # Add pattern
    cmd.append(pattern)
    
    # Add path
    cmd.append(str(p))
    
    # Add file pattern if specified
    if file_pattern:
        cmd.extend(["--include", file_pattern])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in '{path}'"
        
        if result.returncode != 0 and result.stderr:
            return f"Error: {result.stderr.strip()}"
        
        lines = result.stdout.strip().split("\n")
        
        # Filter out empty lines
        lines = [line for line in lines if line.strip()]
        
        if not lines:
            return f"No matches found for pattern '{pattern}' in '{path}'"
        
        total = len(lines)
        
        if total > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (truncated, showing {max_results} of {total} matches)"
        else:
            truncated_msg = ""
        
        output = f"Found {total} match(es) for pattern '{pattern}':\n" + "\n".join(lines) + truncated_msg
        return output
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except FileNotFoundError:
        return "Error: grep command not found"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
