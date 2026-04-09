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
            "Commands: view, create, str_replace, insert, undo_edit, clear_history. "
            "str_replace requires old_str to match exactly and be unique. "
            "Paths must be absolute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit", "clear_history"],
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
    """Track file edit history with memory limits to prevent unbounded growth."""
    
    def __init__(self, max_history_per_file: int = 10) -> None:
        self._history: dict[str, list[str]] = {}
        self._max_history = max_history_per_file

    def add(self, path: str, content: str) -> None:
        """Add content to history, maintaining size limits."""
        if path not in self._history:
            self._history[path] = []
        self._history[path].append(content)
        # Limit history size to prevent memory issues with large files
        if len(self._history[path]) > self._max_history:
            self._history[path].pop(0)  # Remove oldest entry

    def undo(self, path: str) -> str | None:
        """Undo the last edit to a file."""
        if path in self._history and self._history[path]:
            return self._history[path].pop()
        return None
    
    def clear(self, path: str | None = None) -> None:
        """Clear history for a specific file or all files."""
        if path is None:
            self._history.clear()
        elif path in self._history:
            del self._history[path]


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
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if not file_text:
                return "Error: file_text required for create."
            if p.exists():
                return f"Error: {path} already exists. Use str_replace to edit."
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(file_text)
            _history.add(str(p), file_text)
            return f"File created at: {path}"
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str required for str_replace."
            return _replace(p, old_str, new_str or "")
        elif command == "insert":
            if insert_line is None or new_str is None:
                return "Error: insert_line and new_str required for insert."
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
        elif command == "clear_history":
            _history.clear(str(p) if p.exists() else None)
            return f"Edit history cleared for {path}" if p.exists() else "All edit history cleared."
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


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
    if not old_str:
        return "Error: old_str cannot be empty."
    
    content = p.read_text().expandtabs()
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context about what was searched
        # Show first 200 chars of what was searched
        preview = old_str[:200].replace("\n", "\\n")
        if len(old_str) > 200:
            preview += "..."
        # Also show file size info
        file_size = len(content)
        return f"Error: old_str not found in {p} (file size: {file_size} chars). Searched for:\n{preview}"
    if count > 1:
        # Show line numbers where matches occur
        lines = content.split("\n")
        match_lines = []
        for i, line in enumerate(lines, 1):
            if old_str in line:
                match_lines.append(i)
            # Also check across line boundaries
            if i < len(lines):
                combined = line + "\n" + lines[i]
                if old_str in combined and old_str not in line and old_str not in lines[i]:
                    match_lines.append(i)
        return f"Error: old_str appears {count} times in {p} at lines {match_lines[:5]}{'...' if len(match_lines) > 5 else ''}. Make it unique."

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
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
