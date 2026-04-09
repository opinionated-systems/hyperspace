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
            "Commands: find_files (glob patterns), grep (search content), "
            "find_in_files (search multiple files), view (view file contents with line numbers)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files", "view"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path). For 'view' command, this is the file path.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (glob for find_files, regex for grep). For 'view' command, this is optional (use empty string).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to limit search (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50). For 'view' command, this sets max lines to display.",
                },
                "offset": {
                    "type": "integer",
                    "description": "For 'view' command: starting line number (1-indexed, default 1).",
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
    pattern: str,
    file_pattern: str | None = None,
    max_results: int = 50,
    offset: int = 1,
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
            return _find_files(p, pattern, max_results)
        elif command == "grep":
            return _grep(p, pattern, file_pattern, max_results)
        elif command == "find_in_files":
            return _find_in_files(p, pattern, file_pattern, max_results)
        elif command == "view":
            return _view(p, max_results, offset)
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


def _grep(path: Path, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Search for pattern in files using grep."""
    try:
        cmd = ["grep", "-r", "-n", "-I", "--include"]
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


def _find_in_files(path: Path, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Find files containing pattern."""
    try:
        cmd = ["grep", "-r", "-l", "-I", "--include"]
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


def _view(path: Path, max_lines: int = 50, offset: int = 1) -> str:
    """View file contents with line numbers."""
    try:
        if not path.is_file():
            return f"Error: {path} is not a file."
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Convert to 0-indexed
        start_idx = max(0, offset - 1)
        end_idx = min(start_idx + max_lines, total_lines)
        
        if start_idx >= total_lines:
            return f"Error: offset {offset} exceeds file length ({total_lines} lines)"
        
        # Format with line numbers
        result_lines = []
        for i in range(start_idx, end_idx):
            line_num = i + 1  # 1-indexed for display
            line_content = lines[i].rstrip('\n\r')
            result_lines.append(f"{line_num:4d} | {line_content}")
        
        header = f"Viewing {path} (lines {start_idx + 1}-{end_idx} of {total_lines}):\n"
        content = "\n".join(result_lines)
        
        if end_idx < total_lines:
            content += f"\n... ({total_lines - end_idx} more lines)"
        
        return header + content
    except Exception as e:
        return f"Error viewing file: {e}"
