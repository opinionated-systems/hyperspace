"""
Search tool: find content in files using grep-like functionality.

Provides search capabilities for finding text patterns in files,
similar to grep but with safer defaults and better output formatting.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files. "
            "Uses grep-like functionality with safe defaults. "
            "Returns matching lines with file paths and line numbers. "
            "Limited to 100 results to avoid overwhelming output."
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
                    "description": "Directory or file to search in. Must be absolute path.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive. Default: False",
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
    case_sensitive: bool = False,
) -> str:
    """Search for text patterns in files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in (absolute path)
        file_extension: Optional file extension filter
        case_sensitive: Whether search is case-sensitive
        
    Returns:
        Search results with file paths and line numbers
    """
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Limit results
        cmd.extend(["-m", "100"])
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(p))
        
        # Add file extension filter if specified
        if file_extension:
            # Use --include for file extension filtering
            cmd.insert(1, f"--include=*{file_extension}")
        
        # Exclude binary files and hidden directories
        cmd.insert(1, "-I")  # Ignore binary files
        cmd.insert(1, "--exclude-dir=.*")  # Exclude hidden directories
        cmd.insert(1, "--exclude-dir=__pycache__")  # Exclude Python cache
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            output = result.stdout
            # Truncate if too long
            max_len = 10000
            if len(output) > max_len:
                lines = output.split("\n")
                truncated = "\n".join(lines[:50])
                return f"{truncated}\n... [showing first 50 of {len(lines)} matches] ..."
            return output if output else "No matches found."
        elif result.returncode == 1:
            # No matches found (grep returns 1 when no matches)
            return "No matches found."
        else:
            # Error
            return f"Error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30s. Try a more specific pattern or smaller directory."
    except Exception as e:
        return f"Error: {e}"
