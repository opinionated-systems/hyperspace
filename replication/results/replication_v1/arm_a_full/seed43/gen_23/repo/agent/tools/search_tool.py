"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities.
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
            "Search for files and content. "
            "Commands: find_files (by name pattern), grep (search content), "
            "find_in_files (search multiple files). "
            "Useful for locating code or text across the codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (filename pattern or content regex).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern filter (e.g., '*.py' for Python files).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path_allowed(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def _truncate_results(results: list[str], max_results: int = 50) -> str:
    """Truncate results list to max_results."""
    if len(results) > max_results:
        shown = results[:max_results]
        omitted = len(results) - max_results
        return '\n'.join(shown) + f"\n... ({omitted} more results omitted) ..."
    return '\n'.join(results)


def _find_files(path: str, pattern: str, max_results: int = 50) -> str:
    """Find files by name pattern."""
    if not _check_path_allowed(path):
        return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    p = Path(path)
    if not p.exists():
        return f"Error: path {path} does not exist"
    
    results = []
    try:
        # Use find command for efficiency
        cmd = ["find", str(p), "-type", "f", "-name", pattern]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            files = [f for f in result.stdout.strip().split('\n') if f]
            results = files[:max_results]
            
        if not results:
            # Fallback: manual search
            for f in p.rglob(pattern):
                if f.is_file():
                    results.append(str(f))
                    if len(results) >= max_results:
                        break
        
        if not results:
            return f"No files matching '{pattern}' found in {path}"
        
        return f"Found {len(results)} file(s):\n" + _truncate_results(results, max_results)
    
    except subprocess.TimeoutExpired:
        return f"Error: search timed out. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error searching files: {e}"


def _grep(path: str, pattern: str, file_pattern: str | None = None, max_results: int = 50) -> str:
    """Search for content pattern in files."""
    if not _check_path_allowed(path):
        return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    p = Path(path)
    if not p.exists():
        return f"Error: path {path} does not exist"
    
    results = []
    try:
        # Use grep for efficiency
        cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", pattern, str(p)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode in (0, 1):  # 0 = matches found, 1 = no matches
            lines = [line for line in result.stdout.strip().split('\n') if line]
            results = lines[:max_results]
        
        if not results:
            return f"No matches for '{pattern}' in {path}"
        
        return f"Found {len(results)} match(es):\n" + _truncate_results(results, max_results)
    
    except subprocess.TimeoutExpired:
        return f"Error: grep timed out. Try a more specific pattern or narrower path."
    except Exception as e:
        return f"Error in grep: {e}"


def _find_in_files(path: str, pattern: str, file_pattern: str | None = None, max_results: int = 50) -> str:
    """Search for pattern in multiple files with context."""
    if not _check_path_allowed(path):
        return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    p = Path(path)
    if not p.exists():
        return f"Error: path {path} does not exist"
    
    results = []
    count = 0
    
    try:
        # Find files to search
        files = []
        if p.is_file():
            files = [p]
        else:
            glob_pattern = file_pattern or "*"
            for f in p.rglob(glob_pattern):
                if f.is_file() and not any(part.startswith('.') for part in f.parts):
                    files.append(f)
                    if len(files) > 1000:  # Limit files to search
                        break
        
        # Search in each file
        regex = re.compile(pattern, re.IGNORECASE)
        for f in files:
            try:
                content = f.read_text(errors='ignore')
                matches = list(regex.finditer(content))
                if matches:
                    for m in matches:
                        # Get context around match
                        start = max(0, m.start() - 50)
                        end = min(len(content), m.end() + 50)
                        context = content[start:end].replace('\n', ' ')
                        results.append(f"{f}:{m.start()}: ...{context}...")
                        count += 1
                        if count >= max_results:
                            break
                if count >= max_results:
                    break
            except Exception:
                continue
        
        if not results:
            return f"No matches for '{pattern}' in {path}"
        
        return f"Found {count} match(es) in {len(files)} file(s):\n" + '\n'.join(results)
    
    except Exception as e:
        return f"Error searching in files: {e}"


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    if not Path(path).is_absolute():
        return f"Error: {path} is not an absolute path."
    
    if command == "find_files":
        return _find_files(path, pattern, max_results)
    elif command == "grep":
        return _grep(path, pattern, file_pattern, max_results)
    elif command == "find_in_files":
        return _find_in_files(path, pattern, file_pattern, max_results)
    else:
        return f"Error: unknown command {command}"
