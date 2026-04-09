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
    """Execute a file editor command."""
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
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: file_text required for create."
            if p.exists():
                return f"Error: {path} already exists. Use str_replace to edit."
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(file_text)
            _history.add(str(p), file_text)
            return f"File created successfully at: {path} ({len(file_text)} characters)"
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str required for str_replace."
            if new_str is None:
                new_str = ""
            return _replace(p, old_str, new_str)
        elif command == "insert":
            if insert_line is None:
                return "Error: insert_line required for insert."
            if new_str is None:
                new_str = ""
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
        else:
            return f"Error: unknown command {command}"
    except PermissionError as e:
        return f"Error: Permission denied for {path}. {e}"
    except FileNotFoundError as e:
        return f"Error: File or directory not found: {path}. {e}"
    except IsADirectoryError as e:
        return f"Error: {path} is a directory, not a file. {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _view(p: Path, view_range: list[int] | None) -> str:
    if p.is_dir():
        try:
            result = subprocess.run(
                ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\.*"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"Error listing directory {p}: {result.stderr}"
            return f"Files in {p}:\n{_truncate(result.stdout, 5000)}"
        except subprocess.TimeoutExpired:
            return f"Error: Timeout listing directory {p}"
        except Exception as e:
            return f"Error listing directory {p}: {e}"

    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_file():
        return f"Error: {p} is not a regular file."

    try:
        content = p.read_text()
    except UnicodeDecodeError:
        return f"Error: {p} appears to be a binary file and cannot be viewed as text."
    except Exception as e:
        return f"Error reading {p}: {e}"
    
    if view_range:
        lines = content.split("\n")
        try:
            start, end = view_range
            if len(view_range) != 2:
                return f"Error: view_range must be [start, end] with exactly 2 integers."
            if start < 1:
                return f"Error: start line must be >= 1."
            if end == -1:
                end = len(lines)
            if end > len(lines):
                end = len(lines)
            if start > end:
                return f"Error: start ({start}) cannot be greater than end ({end})."
            content = "\n".join(lines[start - 1 : end])
            return _format_output(content, str(p), start)
        except (ValueError, TypeError) as e:
            return f"Error: Invalid view_range format. Must be [start, end] integers. {e}"
    
    # For large files, show a warning
    if len(content) > 100000:
        return _format_output(content, str(p)) + f"\n[Note: File is large ({len(content)} characters)]"
    
    return _format_output(content, str(p))


def _replace(p: Path, old_str: str, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    
    # Handle empty old_str
    if not old_str:
        return "Error: old_str cannot be empty."
    
    content = p.read_text().expandtabs()
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context about what was searched
        content_preview = content[:200] if len(content) > 200 else content
        return f"Error: old_str not found in {p}. Content preview:\n{content_preview}..."
    if count > 1:
        # Show where the matches are
        lines = content.split("\n")
        match_lines = []
        for i, line in enumerate(lines):
            if old_str in line:
                match_lines.append(f"  Line {i+1}: {line[:80]}...")
                if len(match_lines) >= 3:
                    match_lines.append(f"  ... and {count - 3} more matches")
                    break
        return f"Error: old_str appears {count} times in {p}. Make it unique. Matches found at:\n" + "\n".join(match_lines)

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str, 1)  # Only replace first occurrence
    p.write_text(new_content)

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", start + 1)


def _insert(p: Path, line_num: int, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    
    # Handle negative line numbers (Python-style from end)
    content = p.read_text().expandtabs()
    lines = content.split("\n")
    
    if line_num < 0:
        line_num = len(lines) + line_num + 1
    
    if line_num < 0 or line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]. File has {len(lines)} lines."

    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    p.write_text("\n".join(new_lines))

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
    return f"File {p} edited successfully (inserted at line {line_num}). " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))


def _undo(p: Path) -> str:
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}."
    p.write_text(prev)
    return f"Last edit to {p} undone."
