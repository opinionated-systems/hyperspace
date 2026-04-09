"""
Search tool: find patterns in files using grep/ripgrep.

Provides file search capabilities for the agent to locate
specific patterns, functions, or content across the codebase.
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
            "Supports regex patterns and can search recursively. "
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
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively. Default: true.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
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


def _truncate_output(output: str, max_chars: int = 10000) -> str:
    """Truncate output if it exceeds max length."""
    if len(output) > max_chars:
        lines = output.split('\n')
        truncated_lines = []
        current_len = 0
        for line in lines:
            if current_len + len(line) + 1 > max_chars:
                truncated_lines.append(f"... ({len(lines) - len(truncated_lines)} more lines)")
                break
            truncated_lines.append(line)
            current_len += len(line) + 1
        return '\n'.join(truncated_lines)
    return output


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search for patterns in files."""
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No search path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            # Scope check
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        # Build grep command
        cmd = ["grep"]
        
        # Add options
        if not case_sensitive:
            cmd.append("-i")  # Case insensitive
        if recursive:
            cmd.append("-r")
        cmd.append("-n")  # Show line numbers
        cmd.append("-H")  # Show filenames
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(search_path)
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Exclude hidden directories and binary files
        cmd.extend(["--exclude-dir", ".*"])
        cmd.append("-I")  # Ignore binary files
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            lines = result.stdout.strip().split('\n')
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more matches)")
            return _truncate_output('\n'.join(lines))
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}'"
        else:
            # Error
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"
