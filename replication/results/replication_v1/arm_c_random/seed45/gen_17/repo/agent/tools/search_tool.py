"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality and file finding capabilities
to help the meta agent locate code patterns and files efficiently.
"""

from __future__ import annotations

import os
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
            "Search tool for finding files and content. "
            "Commands: grep (search content), find (find files by name). "
            "Searches are scoped to the allowed root directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for grep, glob for find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Filter by file extension (e.g., '.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["command", "pattern"],
        },
    }


def _truncate_output(output: str, max_lines: int = 100) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        # Determine search root
        if path is not None:
            search_root = Path(path)
            if not search_root.is_absolute():
                return f"Error: {path} is not an absolute path."
        else:
            search_root = Path(_ALLOWED_ROOT) if _ALLOWED_ROOT else Path(os.getcwd())

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(search_root))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if command == "grep":
            return _grep(pattern, search_root, file_extension, max_results)
        elif command == "find":
            return _find(pattern, search_root, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, search_root: Path, file_extension: str | None, max_results: int) -> str:
    """Search file contents using grep."""
    if not search_root.exists():
        return f"Error: {search_root} does not exist."

    # Build grep command
    cmd = ["grep", "-r", "-n", "-I", "--include", file_extension or "*"]
    
    # Add pattern and path
    cmd.extend([pattern, str(search_root)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            output = _truncate_output(result.stdout, max_results)
            return f"Found matches:\n{output}"
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit). Try a more specific pattern."
    except FileNotFoundError:
        # Fallback to Python implementation if grep not available
        return _python_grep(pattern, search_root, file_extension, max_results)


def _python_grep(pattern: str, search_root: Path, file_extension: str | None, max_results: int) -> str:
    """Fallback Python implementation of grep."""
    import re
    
    matches = []
    count = 0
    
    try:
        if search_root.is_file():
            files = [search_root]
        else:
            files = list(search_root.rglob("*"))
        
        for f in files:
            if f.is_file() and not any(part.startswith(".") for part in f.parts):
                if file_extension and not str(f).endswith(file_extension):
                    continue
                
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(content.split("\n"), 1):
                        if re.search(pattern, line):
                            matches.append(f"{f}:{i}:{line}")
                            count += 1
                            if count >= max_results:
                                return f"Found matches:\n" + "\n".join(matches)
                except Exception:
                    continue
    except Exception as e:
        return f"Error during search: {e}"
    
    if matches:
        return f"Found matches:\n" + "\n".join(matches)
    return "No matches found."


def _find(pattern: str, search_root: Path, max_results: int) -> str:
    """Find files by name pattern."""
    if not search_root.exists():
        return f"Error: {search_root} does not exist."

    try:
        # Use find command if available
        cmd = ["find", str(search_root), "-name", pattern, "-type", "f"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            files = [f for f in result.stdout.strip().split("\n") if f]
            if not files:
                return "No files found."
            if len(files) > max_results:
                files = files[:max_results]
                return f"Found files:\n" + "\n".join(files) + f"\n... ({len(files) - max_results} more files)"
            return f"Found files:\n" + "\n".join(files)
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit). Try a more specific pattern."
    except FileNotFoundError:
        # Fallback to Python implementation
        return _python_find(pattern, search_root, max_results)


def _python_find(pattern: str, search_root: Path, max_results: int) -> str:
    """Fallback Python implementation of find."""
    import fnmatch
    
    matches = []
    
    try:
        for f in search_root.rglob(pattern):
            if f.is_file() and not any(part.startswith(".") for part in f.parts):
                matches.append(str(f))
                if len(matches) >= max_results:
                    return f"Found files:\n" + "\n".join(matches) + "\n... (more files)"
    except Exception as e:
        return f"Error during search: {e}"
    
    if matches:
        return f"Found files:\n" + "\n".join(matches)
    return "No files found."
