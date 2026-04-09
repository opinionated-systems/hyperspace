"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns within files,
with support for regex and case-insensitive matching.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files. "
            "Supports regex patterns and case-insensitive matching. "
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
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default: true).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["pattern", "path"],
        },
    }


def tool_function(
    pattern: str,
    path: str,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in (absolute path)
        case_sensitive: Whether search is case-sensitive
        max_results: Maximum number of results to return
    
    Returns:
        String with matching lines, file paths, and line numbers
    """
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        results = []
        count = 0
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        
        # Search in single file
        if p.is_file():
            try:
                content = p.read_text()
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if compiled_pattern.search(line):
                        results.append(f"{p}:{i}: {line}")
                        count += 1
                        if count >= max_results:
                            break
            except Exception as e:
                return f"Error reading {p}: {e}"
        
        # Search in directory
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for filename in files:
                    if filename.startswith('.'):
                        continue
                    
                    filepath = Path(root) / filename
                    try:
                        content = filepath.read_text()
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if compiled_pattern.search(line):
                                results.append(f"{filepath}:{i}: {line}")
                                count += 1
                                if count >= max_results:
                                    break
                        if count >= max_results:
                            break
                    except (UnicodeDecodeError, PermissionError):
                        # Skip binary files or files we can't read
                        continue
                    except Exception:
                        continue
                
                if count >= max_results:
                    break
        
        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        # Format output
        header = f"Found {len(results)} match(es) for '{pattern}':\n"
        if count >= max_results:
            header = f"Found {max_results}+ match(es) for '{pattern}' (showing first {max_results}):\n"
        
        # Truncate if too long
        output = '\n'.join(results)
        max_len = 10000
        if len(output) > max_len:
            output = output[:max_len//2] + "\n... [output truncated] ...\n" + output[-max_len//2:]
        
        return header + output
        
    except Exception as e:
        return f"Error: {e}"
