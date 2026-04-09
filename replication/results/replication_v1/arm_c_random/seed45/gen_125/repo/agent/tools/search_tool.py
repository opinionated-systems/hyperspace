"""
Search tool: search for text patterns in files.

Provides grep-like functionality to find text patterns across files.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files. "
            "Uses grep-like functionality to find matches. "
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
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
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
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional extension filter (e.g., '.py')
    
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
        
        matches = []
        files_searched = 0
        
        if p.is_file():
            files = [p]
        else:
            # Find files recursively, but skip common non-source directories
            skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv', 'env'}
            if file_extension:
                files = []
                for f in p.rglob(f"*{file_extension}"):
                    if f.is_file() and not any(skip in f.parts for skip in skip_dirs):
                        files.append(f)
            else:
                files = []
                for f in p.rglob("*"):
                    if f.is_file() and not any(skip in f.parts for skip in skip_dirs):
                        files.append(f)
        
        # Search in each file
        for file_path in files:
            files_searched += 1
            # Limit files searched to prevent timeouts
            if files_searched > 1000:
                break
            try:
                # Skip binary files and very large files
                if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                    continue
                content = file_path.read_text(errors="ignore")
                # Quick check if pattern might be in content
                if not re.search(pattern, content[:10000]):  # Check first 10KB
                    # Pattern not in first 10KB, do full check
                    if len(content) > 10000 and not re.search(pattern, content):
                        continue
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        matches.append(f"{file_path}:{i}: {line[:200]}")
                        if len(matches) >= 100:  # Limit results
                            break
                if len(matches) >= 100:
                    break
            except Exception:
                continue
        
        if not matches:
            return f"No matches found for pattern '{pattern}' in {path} (searched {files_searched} files)"
        
        result = f"Found {len(matches)} matches for pattern '{pattern}' (searched {files_searched} files):\n"
        result += "\n".join(matches[:50])  # Show first 50
        if len(matches) > 50:
            result += f"\n... and {len(matches) - 50} more matches"
        
        return result
        
    except Exception as e:
        return f"Error: {e}"
