"""
Search tool: find patterns in files using grep and find.

Provides utilities for searching code patterns, which is useful
for the meta-agent when modifying the codebase.
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
            "Useful for finding code patterns, function definitions, etc. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (grep-compatible regex).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to limit search (e.g., '*.py').",
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


def tool_function(pattern: str, path: str, file_pattern: str | None = None) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The grep pattern to search for
        path: Directory or file to search in (must be absolute)
        file_pattern: Optional glob pattern to filter files (e.g., "*.py")
    
    Returns:
        Matching lines with file paths and line numbers
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
        
        # Build grep command with fixed strings for literal matching when appropriate
        # Use -F for fixed strings if pattern doesn't look like a regex
        use_fixed = not any(c in pattern for c in "^$.*+?{}[]|()\\")
        
        cmd = ["grep", "-rn"]
        if use_fixed:
            cmd.append("-F")
        cmd.extend(["--include"])
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        cmd.extend([pattern, str(p)])
        
        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Found matches
            output = result.stdout
            if len(output) > 10000:
                output = output[:5000] + "\n... [results truncated] ...\n" + output[-5000:]
            return output
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            # Error
            return f"Error searching: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds. Try a more specific pattern or smaller directory."
    except Exception as e:
        return f"Error: {e}"
