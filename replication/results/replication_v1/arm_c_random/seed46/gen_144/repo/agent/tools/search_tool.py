"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities.
"""

from __future__ import annotations

import os
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
                    "description": "File name pattern (find_files) or search pattern (grep).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern filter for grep (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                    "default": 50,
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


def _check_path(path: str) -> str | None:
    """Check if path is within allowed root. Returns error message or None."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return None


def _truncate_output(output: str, max_lines: int = 100) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def _find_files(path: str, pattern: str, max_results: int) -> str:
    """Find files by name pattern."""
    error = _check_path(path)
    if error:
        return error
    
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"
    
    try:
        # Use find command for efficiency
        cmd = ["find", path, "-type", "f", "-name", pattern]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        files = [f for f in files if f]  # Remove empty strings
        
        if len(files) > max_results:
            files = files[:max_results]
            truncated_msg = f"\n... (truncated, showing {max_results} of {len(files)} results)"
        else:
            truncated_msg = ""
        
        if not files:
            return f"No files matching '{pattern}' found in {path}"
        
        return f"Found {len(files)} file(s):\n" + "\n".join(files) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


def _grep(path: str, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Search for pattern in files."""
    error = _check_path(path)
    if error:
        return error
    
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"
    
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in [0, 1]:
            return f"Error: {result.stderr}"
        
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        lines = [l for l in lines if l]
        
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (truncated, showing {max_results} of {len(lines)} matches)"
        else:
            truncated_msg = ""
        
        if not lines:
            return f"No matches for '{pattern}' in {path}"
        
        return f"Found {len(lines)} match(es):\n" + "\n".join(lines) + truncated_msg
    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


def _find_in_files(path: str, pattern: str, max_results: int) -> str:
    """Search for pattern in specific files (faster than grep for known files)."""
    error = _check_path(path)
    if error:
        return error
    
    try:
        p = Path(path)
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = list(p.rglob("*.py")) + list(p.rglob("*.txt")) + list(p.rglob("*.md"))
        else:
            return f"Error: {path} not found"
        
        matches = []
        for f in files:
            if not f.is_file():
                continue
            try:
                content = f.read_text(errors="ignore")
                if pattern in content:
                    # Find line numbers
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        if pattern in line:
                            matches.append(f"{f}:{i}:{line.strip()}")
                            if len(matches) >= max_results:
                                break
                    if len(matches) >= max_results:
                        break
            except Exception:
                continue
        
        if not matches:
            return f"No matches for '{pattern}' in {path}"
        
        truncated = len(matches) >= max_results
        result = "Found matches:\n" + "\n".join(matches[:max_results])
        if truncated:
            result += f"\n... (truncated at {max_results} results)"
        return result
    except Exception as e:
        return f"Error: {e}"


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    if command == "find_files":
        return _find_files(path, pattern, max_results)
    elif command == "grep":
        return _grep(path, pattern, file_pattern, max_results)
    elif command == "find_in_files":
        return _find_in_files(path, pattern, max_results)
    else:
        return f"Error: unknown command {command}"
