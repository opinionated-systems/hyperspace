"""
Search tool: search for patterns across files in the repository.

Provides grep-like functionality to find code patterns, function definitions,
and references across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search_files",
        "description": (
            "Search for patterns across files in the repository. "
            "Uses grep/ripgrep for efficient searching. "
            "Returns matching files with line numbers and context."
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
                    "description": "Directory to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', '*.js'). Default: all files.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
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
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern across files.
    
    Args:
        pattern: The search pattern (regex supported)
        path: Directory to search in
        file_pattern: Optional file pattern filter (e.g., '*.py')
        max_results: Maximum number of matches to return
    
    Returns:
        Search results with file paths, line numbers, and matching lines
    """
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        if not p.is_dir():
            return f"Error: {path} is not a directory."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        # Try ripgrep first, fall back to grep
        results = []
        
        # Build command
        if _has_ripgrep():
            cmd = ["rg", "-n", "--color=never", "-C", "2"]
            if file_pattern:
                cmd.extend(["-g", file_pattern])
            cmd.extend([pattern, str(p)])
        else:
            cmd = ["grep", "-rn", "-C", "2"]
            if file_pattern:
                # grep doesn't support file patterns directly, use find
                pass
            cmd.extend([pattern, str(p)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode not in [0, 1]:  # 0 = matches found, 1 = no matches
                return f"Search error: {result.stderr}"
            
            lines = result.stdout.strip().split("\n")
            
            # Parse and limit results
            matches = []
            for line in lines[:max_results * 3]:  # Account for context lines
                if line.strip():
                    matches.append(line)
            
            if not matches:
                return f"No matches found for pattern '{pattern}' in {path}"
            
            # Truncate if too long
            output = "\n".join(matches[:max_results * 3])
            if len(matches) > max_results * 3:
                output += f"\n... [truncated, showing first {max_results} matches]"
            
            return f"Search results for '{pattern}':\n{output}"
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds"
        except FileNotFoundError:
            # Neither ripgrep nor grep available, use Python fallback
            return _python_search(p, pattern, file_pattern, max_results)
            
    except Exception as e:
        return f"Error: {e}"


def _has_ripgrep() -> bool:
    """Check if ripgrep is available."""
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _python_search(
    path: Path,
    pattern: str,
    file_pattern: str | None,
    max_results: int,
) -> str:
    """Fallback search using Python's re module."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    matches = []
    count = 0
    
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for filename in files:
            if filename.startswith("."):
                continue
            
            if file_pattern and not _match_pattern(filename, file_pattern):
                continue
            
            filepath = Path(root) / filename
            
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    lines = content.split("\n")
                    
                    for i, line in enumerate(lines, 1):
                        if regex.search(line):
                            matches.append(f"{filepath}:{i}:{line.strip()}")
                            count += 1
                            
                            if count >= max_results:
                                break
                    
                    if count >= max_results:
                        break
            except (IOError, OSError):
                continue
        
        if count >= max_results:
            break
    
    if not matches:
        return f"No matches found for pattern '{pattern}' in {path}"
    
    output = "\n".join(matches)
    if count >= max_results:
        output += f"\n... [showing first {max_results} matches]"
    
    return f"Search results for '{pattern}':\n{output}"


def _match_pattern(filename: str, pattern: str) -> bool:
    """Simple glob-style pattern matching."""
    import fnmatch
    return fnmatch.fnmatch(filename, pattern)
