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
            "Commands: find_files (glob patterns), grep (search content with context), "
            "find_in_files (find files containing pattern), view_file (view specific file with line numbers)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files", "view_file"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in or file to view (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (glob for find_files, regex for grep). Not needed for view_file.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to limit search (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive (default: True).",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines for grep (default: 2).",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Line range [start, end] for view_file command.",
                },
            },
            "required": ["command", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_output(lines: list[str], max_lines: int = 50) -> str:
    """Truncate output to max_lines."""
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more results)"
    return "\n".join(lines)


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = True,
    context_lines: int = 2,
    view_range: list[int] | None = None,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if command == "find_files":
            if pattern is None:
                return "Error: pattern required for find_files command."
            return _find_files(p, pattern, max_results)
        elif command == "grep":
            if pattern is None:
                return "Error: pattern required for grep command."
            return _grep(p, pattern, file_pattern, max_results, case_sensitive, context_lines)
        elif command == "find_in_files":
            if pattern is None:
                return "Error: pattern required for find_in_files command."
            return _find_in_files(p, pattern, file_pattern, max_results, case_sensitive)
        elif command == "view_file":
            return _view_file(p, view_range)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(path: Path, pattern: str, max_results: int) -> str:
    """Find files matching glob pattern."""
    try:
        matches = list(path.rglob(pattern))
        # Filter to files only
        files = [str(m.relative_to(path)) for m in matches if m.is_file()]
        files.sort()
        
        if not files:
            return f"No files matching '{pattern}' found in {path}"
        
        output = _truncate_output(files, max_results)
        return f"Found {len(files)} files matching '{pattern}':\n{output}"
    except Exception as e:
        return f"Error finding files: {e}"


def _grep(path: Path, pattern: str, file_pattern: str | None, max_results: int, case_sensitive: bool = True, context_lines: int = 2) -> str:
    """Search for pattern in files using grep with context."""
    try:
        cmd = ["grep", "-r", "-n", "-I"]
        if not case_sensitive:
            cmd.append("-i")
        # Add context lines
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        cmd.append("--include")
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        cmd.extend(["-e", pattern, str(path)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 1:
            return f"No matches for '{pattern}' in {path}"
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        
        lines = result.stdout.strip().split("\n")
        if not lines or lines == [""]:
            return f"No matches for '{pattern}' in {path}"
        
        output = _truncate_output(lines, max_results)
        return f"Found {len(lines)} matches for '{pattern}':\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30s"
    except Exception as e:
        return f"Error searching: {e}"


def _view_file(path: Path, view_range: list[int] | None) -> str:
    """View a file with line numbers, optionally with a line range."""
    try:
        if path.is_dir():
            return f"Error: {path} is a directory, not a file."
        
        if not path.exists():
            return f"Error: {path} does not exist."
        
        content = path.read_text()
        lines = content.split("\n")
        
        if view_range:
            start, end = view_range
            if start < 1:
                start = 1
            if end == -1 or end > len(lines):
                end = len(lines)
            lines = lines[start - 1:end]
            init_line = start
        else:
            init_line = 1
        
        # Format with line numbers
        numbered = [f"{i + init_line:6}\t{line}" for i, line in enumerate(lines)]
        output = "\n".join(numbered)
        
        # Truncate if too long
        if len(output) > 10000:
            half = 5000
            output = output[:half] + "\n... [output truncated] ...\n" + output[-half:]
        
        return f"Here's the result of running `cat -n` on {path}:\n{output}\n"
    except Exception as e:
        return f"Error viewing file: {e}"


def _find_in_files(path: Path, pattern: str, file_pattern: str | None, max_results: int, case_sensitive: bool = True) -> str:
    """Find files containing pattern."""
    try:
        cmd = ["grep", "-r", "-l", "-I"]
        if not case_sensitive:
            cmd.append("-i")
        cmd.append("--include")
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        cmd.extend(["-e", pattern, str(path)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 1:
            return f"No files containing '{pattern}' in {path}"
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        
        files = result.stdout.strip().split("\n")
        files = [f for f in files if f]  # Remove empty strings
        files.sort()
        
        if not files:
            return f"No files containing '{pattern}' in {path}"
        
        output = _truncate_output(files, max_results)
        return f"Found {len(files)} files containing '{pattern}':\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30s"
    except Exception as e:
        return f"Error searching: {e}"
