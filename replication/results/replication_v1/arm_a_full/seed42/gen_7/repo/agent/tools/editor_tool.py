"""
File editor tool: view, create, str_replace, insert, undo_edit.

Reimplemented from facebookresearch/HyperAgents agent/tools/edit.py.
Same commands, same validation, same history tracking.
Enhanced with better error handling and input validation.
"""

from __future__ import annotations

import os
import subprocess
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum file size for reading (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def tool_info() -> dict:
    return {
        "name": "editor",
        "description": (
            "File editor for viewing, creating, and editing files. "
            "Commands: view, create, str_replace, insert, undo_edit. "
            "str_replace requires old_str to match exactly and be unique. "
            "Paths must be absolute. "
            "Files larger than 10MB cannot be viewed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "description": "The command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory.",
                },
                "file_text": {
                    "type": "string",
                    "description": "Content for create command.",
                },
                "old_str": {
                    "type": "string",
                    "description": "String to replace (str_replace).",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string (str_replace/insert).",
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Line number to insert after (insert).",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Line range [start, end] for view.",
                },
            },
            "required": ["command", "path"],
        },
    }


class _FileHistory:
    def __init__(self) -> None:
        self._history: dict[str, list[str]] = {}

    def add(self, path: str, content: str) -> None:
        self._history.setdefault(path, []).append(content)

    def undo(self, path: str) -> str | None:
        if path in self._history and self._history[path]:
            return self._history[path].pop()
        return None


_history = _FileHistory()


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def _format_output(content: str, path: str, init_line: int = 1) -> str:
    content = _truncate(content).expandtabs()
    numbered = [f"{i + init_line:6}\t{line}" for i, line in enumerate(content.split("\n"))]
    return f"Here's the result of running `cat -n` on {path}:\n" + "\n".join(numbered) + "\n"


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict editor operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _validate_command(command: str) -> tuple[bool, str]:
    """Validate the command parameter.
    
    Returns:
        (is_valid, error_message)
    """
    valid_commands = {"view", "create", "str_replace", "insert", "undo_edit"}
    if not isinstance(command, str):
        return False, f"Command must be a string, got {type(command).__name__}"
    if command not in valid_commands:
        return False, f"Unknown command: {command}. Valid commands: {sorted(valid_commands)}"
    return True, ""


def _validate_path(path: str) -> tuple[bool, str, Path | None]:
    """Validate the path parameter.
    
    Returns:
        (is_valid, error_message, path_object)
    """
    if not isinstance(path, str):
        return False, f"Path must be a string, got {type(path).__name__}", None
    
    if not path:
        return False, "Path cannot be empty", None
    
    try:
        p = Path(path)
    except Exception as e:
        return False, f"Invalid path: {e}", None
    
    if not p.is_absolute():
        return False, f"{path} is not an absolute path", None
    
    return True, "", p


def tool_function(
    command: str,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
) -> str:
    """Execute a file editor command with enhanced validation."""
    # Validate command
    is_valid, error_msg = _validate_command(command)
    if not is_valid:
        return f"Error: {error_msg}"
    
    # Validate path
    is_valid, error_msg, p = _validate_path(path)
    if not is_valid:
        return f"Error: {error_msg}"
    
    # Scope check: only allow operations within the allowed root
    if _ALLOWED_ROOT is not None:
        try:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                logger.warning(f"Access denied: {resolved} outside {_ALLOWED_ROOT}")
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        except Exception as e:
            return f"Error: Path resolution failed - {e}"

    try:
        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: file_text required for create."
            if p.exists():
                return f"Error: {path} already exists. Use str_replace to edit."
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(file_text)
                _history.add(str(p), file_text)
                logger.info(f"Created file: {path}")
                return f"File created at: {path}"
            except OSError as e:
                return f"Error: Failed to create file - {e}"
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str required for str_replace."
            return _replace(p, old_str, new_str or "")
        elif command == "insert":
            if insert_line is None:
                return "Error: insert_line required for insert."
            if new_str is None:
                return "Error: new_str required for insert."
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
        else:
            return f"Error: unknown command {command}"
    except PermissionError as e:
        logger.error(f"Permission denied for {path}: {e}")
        return f"Error: Permission denied - {e}"
    except OSError as e:
        logger.error(f"OS error for {path}: {e}")
        return f"Error: OS error - {e}"
    except Exception as e:
        logger.error(f"Unexpected error in {command} on {path}: {e}")
        return f"Error: {type(e).__name__}: {e}"


def _view(p: Path, view_range: list[int] | None) -> str:
    if p.is_dir():
        try:
            result = subprocess.run(
                ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return f"Error: find command failed - {result.stderr}"
            return f"Files in {p}:\n{_truncate(result.stdout, 5000)}"
        except subprocess.TimeoutExpired:
            return f"Error: Directory listing timed out"
        except Exception as e:
            return f"Error: Failed to list directory - {e}"

    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_file():
        return f"Error: {p} is not a file."

    # Check file size
    try:
        file_size = p.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return f"Error: File {p} is too large ({file_size} bytes, max: {MAX_FILE_SIZE}). Use bash tool to view specific parts."
    except OSError as e:
        return f"Error: Cannot stat file - {e}"

    try:
        content = p.read_text()
    except UnicodeDecodeError:
        return f"Error: {p} appears to be a binary file and cannot be viewed as text."
    except OSError as e:
        return f"Error: Cannot read file - {e}"
    
    if view_range:
        lines = content.split("\n")
        start, end = view_range
        # Validate range
        if start < 1:
            return f"Error: view_range start must be >= 1, got {start}"
        if end != -1 and end < start:
            return f"Error: view_range end ({end}) must be >= start ({start})"
        if end == -1:
            end = len(lines)
        end = min(end, len(lines))  # Clamp to file length
        content = "\n".join(lines[start - 1 : end])
        return _format_output(content, str(p), start)
    return _format_output(content, str(p))


def _replace(p: Path, old_str: str, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    content = p.read_text().expandtabs()
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        return f"Error: old_str not found in {p}"
    if count > 1:
        return f"Error: old_str appears {count} times in {p}. Make it unique."

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
    p.write_text(new_content)

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited. " + _format_output(snippet, f"snippet of {p}", start + 1)


def _insert(p: Path, line_num: int, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    content = p.read_text().expandtabs()
    lines = content.split("\n")

    if line_num < 0 or line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]"

    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    p.write_text("\n".join(new_lines))

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
    return f"File {p} edited. " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))


def _undo(p: Path) -> str:
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}."
    p.write_text(prev)
    return f"Last edit to {p} undone."
