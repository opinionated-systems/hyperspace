"""
File editor tool: view, create, str_replace, insert, undo_edit.

Reimplemented from facebookresearch/HyperAgents agent/tools/edit.py.
Same commands, same validation, same history tracking.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "editor",
        "description": (
            "File editor for viewing, creating, and editing files. "
            "Commands: view, create, str_replace, insert, undo_edit. "
            "str_replace requires old_str to match exactly and be unique. "
            "Paths must be absolute."
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


def tool_function(
    command: str,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
) -> str:
    """Execute a file editor command.
    
    Enhanced with better error messages and validation.
    """
    try:
        # Validate command
        valid_commands = ["view", "create", "str_replace", "insert", "undo_edit"]
        if command not in valid_commands:
            return f"Error: unknown command '{command}'. Valid commands: {', '.join(valid_commands)}"
        
        # Validate path
        if not path:
            return "Error: path is required."
            
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path. Please provide an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Path '{path}' is outside allowed root '{_ALLOWED_ROOT}'. Operations are restricted to the allowed root directory."

        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: file_text is required for create command."
            if p.exists():
                return f"Error: cannot create file - {path} already exists. Use 'str_replace' to edit existing files."
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(file_text)
                _history.add(str(p), file_text)
                return f"File created successfully at: {path}"
            except PermissionError:
                return f"Error: permission denied when creating file at {path}."
            except OSError as e:
                return f"Error: failed to create file at {path}: {e}"
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str is required for str_replace command."
            return _replace(p, old_str, new_str)
        elif command == "insert":
            if insert_line is None:
                return "Error: insert_line is required for insert command."
            if new_str is None:
                return "Error: new_str is required for insert command."
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
    except PermissionError as e:
        return f"Error: permission denied for {path}: {e}"
    except FileNotFoundError as e:
        return f"Error: file or directory not found: {path}"
    except IsADirectoryError as e:
        return f"Error: expected a file but found a directory: {path}"
    except NotADirectoryError as e:
        return f"Error: expected a directory but found a file: {path}"
    except Exception as e:
        return f"Error: unexpected error - {type(e).__name__}: {e}"


def _view(p: Path, view_range: list[int] | None) -> str:
    """View file or directory contents."""
    if p.is_dir():
        try:
            result = subprocess.run(
                ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"Error listing directory {p}: {result.stderr}"
            output = result.stdout if result.stdout else "(empty directory)"
            return f"Files in {p}:\n{_truncate(output, 5000)}"
        except subprocess.TimeoutExpired:
            return f"Error: timeout listing directory {p} (too many files)"
        except Exception as e:
            return f"Error listing directory {p}: {e}"

    if not p.exists():
        return f"Error: {p} does not exist."

    try:
        content = p.read_text()
    except UnicodeDecodeError:
        return f"Error: {p} appears to be a binary file and cannot be displayed as text."
    except PermissionError:
        return f"Error: permission denied reading {p}."
    except Exception as e:
        return f"Error reading {p}: {e}"
    
    if view_range:
        lines = content.split("\n")
        start, end = view_range
        
        # Validate range
        if start < 1:
            return f"Error: view_range start ({start}) must be >= 1"
        if end != -1 and end < start:
            return f"Error: view_range end ({end}) must be >= start ({start})"
        
        if end == -1:
            end = len(lines)
        end = min(end, len(lines))  # Clamp to file length
        content = "\n".join(lines[start - 1 : end])
        return _format_output(content, str(p), start)
    return _format_output(content, str(p))


def _replace(p: Path, old_str: str, new_str: str | None) -> str:
    """Replace old_str with new_str in file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_file():
        return f"Error: {p} is not a file."
    
    # Handle None for new_str
    if new_str is None:
        new_str = ""
    
    try:
        content = p.read_text().expandtabs()
    except UnicodeDecodeError:
        return f"Error: {p} appears to be a binary file and cannot be edited as text."
    except PermissionError:
        return f"Error: permission denied reading {p}."
    except Exception as e:
        return f"Error reading {p}: {e}"
    
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context about what was searched
        preview = content[:200] + "..." if len(content) > 200 else content
        return f"Error: old_str not found in {p}. The string you provided does not match any content in the file. Content preview:\n{preview}"
    if count > 1:
        # Show where the matches are
        lines = []
        idx = 0
        for _ in range(min(count, 5)):  # Show up to 5 matches
            idx = content.find(old_str, idx)
            line_num = content[:idx].count("\n") + 1
            lines.append(str(line_num))
            idx += len(old_str)
        return f"Error: old_str appears {count} times in {p} at lines: {', '.join(lines)}. The string must be unique - try including more context to make it unique."

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
    
    try:
        p.write_text(new_content)
    except PermissionError:
        return f"Error: permission denied writing to {p}."
    except Exception as e:
        return f"Error writing to {p}: {e}"

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", start + 1)


def _insert(p: Path, line_num: int, new_str: str | None) -> str:
    """Insert new_str after line_num in file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_file():
        return f"Error: {p} is not a file."
    
    # Handle None for new_str
    if new_str is None:
        new_str = ""
    
    try:
        content = p.read_text().expandtabs()
    except UnicodeDecodeError:
        return f"Error: {p} appears to be a binary file and cannot be edited as text."
    except PermissionError:
        return f"Error: permission denied reading {p}."
    except Exception as e:
        return f"Error reading {p}: {e}"
    
    lines = content.split("\n")

    if line_num < 0 or line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]. File has {len(lines)} lines."

    _history.add(str(p), content)
    new_str_expanded = new_str.expandtabs()
    new_lines = lines[:line_num] + new_str_expanded.split("\n") + lines[line_num:]
    
    try:
        p.write_text("\n".join(new_lines))
    except PermissionError:
        return f"Error: permission denied writing to {p}."
    except Exception as e:
        return f"Error writing to {p}: {e}"

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str_expanded.count("\n")])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))


def _undo(p: Path) -> str:
    """Undo the last edit to a file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}. Cannot undo - no previous versions stored."
    
    try:
        p.write_text(prev)
        return f"Last edit to {p} undone successfully. File restored to previous version."
    except PermissionError:
        return f"Error: permission denied writing to {p}."
    except Exception as e:
        return f"Error undoing edit to {p}: {e}"
