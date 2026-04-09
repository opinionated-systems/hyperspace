"""
Search tool: search for patterns in files within the repository.

Provides grep-like functionality to find text patterns across multiple files.
Useful for exploring codebases and finding relevant code sections.
"""

from __future__ import annotations

import os
import re
import subprocess


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for a pattern in files within the repository. "
            "Uses grep-like functionality to find text patterns. "
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
                    "description": "Directory or file to search in (default: current directory).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: false).",
                },
            },
            "required": ["pattern"],
        },
    }


_MAX_RESULTS = 100  # Maximum number of matches to return
_MAX_OUTPUT_SIZE = 30000  # Maximum output size before truncation


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional extension filter (e.g., '.py')
        case_sensitive: Whether search is case sensitive
    
    Returns:
        Matching lines with file paths and line numbers
    """
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(path)
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Exclude binary files and common non-source directories
        cmd.extend([
            "--exclude-dir", "__pycache__",
            "--exclude-dir", ".git",
            "--exclude-dir", "node_modules",
            "--exclude-dir", "venv",
            "--exclude-dir", ".venv",
            "--exclude-dir", "env",
            "--exclude", "*.pyc",
            "--exclude", "*.pyo",
        ])
        
        # Run grep
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns exit code 1 when no matches found
        if result.returncode not in (0, 1):
            return f"Error: grep failed with exit code {result.returncode}"
        
        output = result.stdout
        
        if not output:
            return f"No matches found for pattern '{pattern}'"
        
        # Count lines and truncate if needed
        lines = output.strip().split("\n")
        if len(lines) > _MAX_RESULTS:
            lines = lines[:_MAX_RESULTS]
            lines.append(f"\n... [truncated - showing first {_MAX_RESULTS} of {len(lines)}+ matches] ...")
        
        output = "\n".join(lines)
        
        # Final size check
        if len(output) > _MAX_OUTPUT_SIZE:
            output = output[:_MAX_OUTPUT_SIZE // 2] + f"\n... [output truncated - {len(output)} chars total] ...\n" + output[-_MAX_OUTPUT_SIZE // 2:]
        
        return output
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
