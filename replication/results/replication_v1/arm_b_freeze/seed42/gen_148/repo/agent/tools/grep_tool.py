"""
Grep tool: search for patterns within file contents.

Provides content-based file searching to complement the file_search_tool
(which searches by filename). This enables finding code patterns, TODOs,
function definitions, and specific text within files.
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
            "Search for patterns within file contents using grep. "
            "Returns matching lines with file paths and line numbers. "
            "Supports regex patterns. Useful for finding code patterns, "
            "TODOs, function definitions, or specific text in files."
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
                    "description": "File glob pattern to filter files (e.g., '*.py', '*.js'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return. Default: 50.",
                },
            },
            "required": ["pattern"],
        },
    }


def _validate_path(path: str, allowed_root: str | None = None) -> Path:
    """Validate and resolve the search path."""
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
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for pattern in file contents.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in
        file_pattern: Optional file glob pattern (e.g., '*.py')
        case_sensitive: Whether search is case sensitive
        max_results: Maximum number of matches to return
    
    Returns:
        Formatted string with matches or error message
    """
    try:
        # Validate path
        target_path = _validate_path(path)
        
        if not target_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        # Add case sensitivity flag
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(target_path))
        
        # Add file pattern if specified
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Exclude binary files and common non-source directories
        cmd.extend([
            "--binary-files=without-match",
            "--exclude-dir=.git",
            "--exclude-dir=__pycache__",
            "--exclude-dir=node_modules",
            "--exclude-dir=.venv",
            "--exclude-dir=venv",
        ])
        
        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Process results
        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        if not lines or lines == [""]:
            return f"No matches found for pattern '{pattern}'"
        
        # Limit results
        total_matches = len(lines)
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated = True
        else:
            truncated = False
        
        # Format output
        output_lines = []
        for line in lines:
            if line:
                output_lines.append(line)
        
        result_text = "\n".join(output_lines)
        
        if truncated:
            result_text += f"\n... ({total_matches - max_results} more matches truncated)"
        
        return result_text
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds"
    except re.error as e:
        return f"Error: Invalid regex pattern - {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
