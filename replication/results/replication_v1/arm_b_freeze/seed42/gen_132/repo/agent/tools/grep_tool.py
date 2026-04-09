"""
Grep tool: search for patterns in files across the codebase.

Provides fast text search capabilities to find code patterns,
function definitions, imports, and more across multiple files.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "grep",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns and can search recursively. "
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
                    "description": "Directory or file to search in. Defaults to current directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories. Default: true",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py'). Default: all files",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case sensitive search. Default: false",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50",
                },
            },
            "required": ["pattern"],
        },
    }


def _validate_path(path: str, allowed_root: str | None = None) -> Path:
    """Validate and resolve path within allowed root."""
    target = Path(path).resolve()
    if allowed_root:
        root = Path(allowed_root).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ValueError(f"Path {path} is outside allowed root {allowed_root}")
    return target


def tool_function(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        recursive: Whether to search subdirectories
        file_pattern: Glob pattern for files (e.g., '*.py')
        case_sensitive: Whether search is case sensitive
        max_results: Maximum results to return
    
    Returns:
        Formatted string with matches
    """
    try:
        target_path = Path(path).resolve()
        
        if not target_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        # Build grep command
        cmd = ["grep", "-n"]  # Always show line numbers
        
        if not case_sensitive:
            cmd.append("-i")  # Case insensitive
        
        if recursive and target_path.is_dir():
            cmd.append("-r")
        
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Add pattern and path
        cmd.extend([pattern, str(target_path)])
        
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
            return f"No matches found for pattern '{pattern}'"
        
        # Limit results
        total_matches = len(lines)
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
                    # Make path relative if possible
                    try:
                        rel_path = Path(file_path).relative_to(Path.cwd())
                        file_path = str(rel_path)
                    except ValueError:
                        pass
                    formatted.append(f"{file_path}:{line_num}: {content}")
                else:
                    formatted.append(line)
            else:
                formatted.append(line)
        
        output = "\n".join(formatted)
        
        if truncated:
            output += f"\n... ({total_matches - max_results} more matches truncated)"
        
        return output
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"
