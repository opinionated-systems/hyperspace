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
    command: str | None = None,
    path: str | None = None,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
) -> str:
    """Execute a file editor command with enhanced validation."""
    # Validate required parameters
    if command is None:
        return "Error: 'command' parameter is required"
    
    if not isinstance(command, str):
        return f"Error: 'command' must be a string, got {type(command).__name__}"
    
    command = command.strip().lower()
    valid_commands = ["view", "create", "str_replace", "insert", "undo_edit"]
    if command not in valid_commands:
        return f"Error: unknown command '{command}'. Valid commands: {', '.join(valid_commands)}"
    
    if path is None:
        return f"Error: 'path' parameter is required for command '{command}'"
    
    if not isinstance(path, str):
        return f"Error: 'path' must be a string, got {type(path).__name__}"
    
    path = path.strip()
    if not path:
        return "Error: 'path' cannot be empty"
    
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path. Please provide a full path starting with /"

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Path '{path}' is outside allowed root '{_ALLOWED_ROOT}'"

        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: 'file_text' parameter is required for create command"
            if p.exists():
                return f"Error: {path} already exists. Use str_replace to edit existing files."
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                # Validate file_text is a string
                if not isinstance(file_text, str):
                    file_text = str(file_text)
                p.write_text(file_text, encoding='utf-8')
                _history.add(str(p), file_text)
                return f"File created successfully at: {path}"
            except Exception as e:
                return f"Error creating file: {e}"
        elif command == "str_replace":
            if old_str is None:
                return "Error: 'old_str' parameter is required for str_replace command"
            if not isinstance(old_str, str):
                return f"Error: 'old_str' must be a string, got {type(old_str).__name__}"
            return _replace(p, old_str, new_str or "")
        elif command == "insert":
            if insert_line is None:
                return "Error: 'insert_line' parameter is required for insert command"
            if new_str is None:
                return "Error: 'new_str' parameter is required for insert command"
            if not isinstance(insert_line, int):
                try:
                    insert_line = int(insert_line)
                except (ValueError, TypeError):
                    return f"Error: 'insert_line' must be an integer, got {type(insert_line).__name__}"
            if not isinstance(new_str, str):
                new_str = str(new_str)
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        import traceback
        return f"Error in editor tool: {e}\n{traceback.format_exc()}"


def _view(p: Path, view_range: list[int] | None) -> str:
    if p.is_dir():
        result = subprocess.run(
            ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
            capture_output=True, text=True,
        )
        return f"Files in {p}:\n{_truncate(result.stdout, 5000)}"

    if not p.exists():
        return f"Error: {p} does not exist."

    content = p.read_text()
    if view_range:
        lines = content.split("\n")
        start, end = view_range
        if end == -1:
            end = len(lines)
        content = "\n".join(lines[start - 1 : end])
        return _format_output(content, str(p), start)
    return _format_output(content, str(p))


def _replace(p: Path, old_str: str, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_file():
        return f"Error: {p} is not a file (might be a directory)."
    
    # Validate inputs
    if old_str is None:
        return "Error: old_str is required for str_replace"
    
    try:
        content = p.read_text(encoding='utf-8').expandtabs()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            content = p.read_text(encoding='latin-1').expandtabs()
        except Exception as e:
            return f"Error reading file {p}: {e}"
    except Exception as e:
        return f"Error reading file {p}: {e}"
    
    old_str = old_str.expandtabs()
    new_str = (new_str or "").expandtabs()

    # Check for exact match first
    count = content.count(old_str)
    if count == 0:
        # Try with stripped whitespace for more flexible matching
        stripped_old = old_str.strip()
        if stripped_old and stripped_old != old_str:
            # Try to find the stripped version
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if stripped_old in line:
                    return f"Error: old_str not found exactly. Found similar on line {i+1}: {line[:80]}..."
        
        # Check if it's a line ending issue
        if old_str.rstrip() != old_str:
            # Has trailing whitespace
            stripped_count = content.count(old_str.rstrip())
            if stripped_count > 0:
                return f"Error: old_str not found exactly. Found {stripped_count} matches without trailing whitespace. Check your old_str for trailing spaces/tabs."
        
        return f"Error: old_str not found in {p}"
    
    if count > 1:
        # Show where the duplicates are
        lines = content.split("\n")
        locations = []
        for i, line in enumerate(lines):
            if old_str in line:
                locations.append(f"line {i+1}")
                if len(locations) >= 5:  # Show more context
                    break
        return f"Error: old_str appears {count} times in {p} (at {', '.join(locations)}...). Make it unique by including more context (more lines before and after)."

    # Save to history before making changes
    _history.add(str(p), content)
    
    try:
        new_content = content.replace(old_str, new_str, 1)  # Only replace first occurrence
        p.write_text(new_content, encoding='utf-8')
    except Exception as e:
        return f"Error writing file {p}: {e}"

    # Show context around edit
    try:
        line_num = content.split(old_str)[0].count("\n")
        start = max(0, line_num - 4)
        end = line_num + 4 + new_str.count("\n")
        snippet = "\n".join(new_content.split("\n")[start : end + 1])
        return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", start + 1)
    except Exception as e:
        return f"File {p} edited successfully, but could not generate preview: {e}"


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
