"""
File editor tool: view, create, str_replace, insert, undo_edit.

Reimplemented from facebookresearch/HyperAgents agent/tools/edit.py.
Same commands, same validation, same history tracking.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class EditStatus(Enum):
    """Status of an edit operation."""
    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"
    VALIDATION_ERROR = "validation_error"


@dataclass
class EditResult:
    """Result of an edit operation."""
    status: EditStatus
    message: str
    path: Optional[Path] = None
    lines_affected: int = 0


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
            result = _create(p, file_text)
            return result.message
        elif command == "str_replace":
            result = _replace(p, old_str, new_str or "")
            return result.message
        elif command == "insert":
            if insert_line is None or new_str is None:
                return "Error: insert_line and new_str required for insert."
            result = _insert(p, insert_line, new_str)
            return result.message
        elif command == "undo_edit":
            result = _undo(p)
            return result.message
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


def _create(p: Path, file_text: str | None) -> EditResult:
    """Create a new file with the given content."""
    if not file_text:
        return EditResult(
            status=EditStatus.VALIDATION_ERROR,
            message="Error: file_text required for create.",
            path=p
        )
    
    if p.exists():
        return EditResult(
            status=EditStatus.VALIDATION_ERROR,
            message=f"Error: {p} already exists. Use str_replace to edit.",
            path=p
        )
    
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text, encoding='utf-8')
        _history.add(str(p), file_text)
        return EditResult(
            status=EditStatus.SUCCESS,
            message=f"File created at: {p}",
            path=p,
            lines_affected=len(file_text.splitlines())
        )
    except Exception as e:
        return EditResult(
            status=EditStatus.ERROR,
            message=f"Error creating file: {e}",
            path=p
        )


def _replace(p: Path, old_str: str | None, new_str: str) -> EditResult:
    """Replace old_str with new_str in the file."""
    if not p.exists():
        return EditResult(
            status=EditStatus.NOT_FOUND,
            message=f"Error: {p} does not exist.",
            path=p
        )
    
    if not old_str:
        return EditResult(
            status=EditStatus.VALIDATION_ERROR,
            message="Error: old_str cannot be empty.",
            path=p
        )
    
    # Check if file is binary
    try:
        content = p.read_text(encoding='utf-8').expandtabs()
    except UnicodeDecodeError:
        return EditResult(
            status=EditStatus.ERROR,
            message=f"Error: {p} appears to be a binary file and cannot be edited as text.",
            path=p
        )
    except Exception as e:
        return EditResult(
            status=EditStatus.ERROR,
            message=f"Error reading {p}: {type(e).__name__}: {e}",
            path=p
        )
    
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context about what was searched
        preview = old_str[:200].replace("\n", "\\n")
        if len(old_str) > 200:
            preview += "..."
        file_size = len(content)
        suggestion = ""
        if len(old_str) > 10:
            partial = old_str[:50]
            if partial in content:
                suggestion = f"\nHint: The first 50 characters were found. Check for whitespace or character differences."
        return EditResult(
            status=EditStatus.VALIDATION_ERROR,
            message=f"Error: old_str not found in {p} (file size: {file_size} chars). Searched for:\n{preview}{suggestion}",
            path=p
        )
    
    if count > 1:
        # Show line numbers where matches occur
        lines = content.split("\n")
        match_lines = []
        for i, line in enumerate(lines, 1):
            if old_str in line:
                match_lines.append(i)
            if i < len(lines):
                combined = line + "\n" + lines[i]
                if old_str in combined and old_str not in line and old_str not in lines[i]:
                    match_lines.append(i)
        return EditResult(
            status=EditStatus.VALIDATION_ERROR,
            message=f"Error: old_str appears {count} times in {p} at lines {match_lines[:5]}{'...' if len(match_lines) > 5 else ''}. Make it unique.",
            path=p
        )

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str, 1)  # Only replace first occurrence
    
    # Validate the write operation
    try:
        p.write_text(new_content, encoding='utf-8')
        # Verify the write succeeded
        verify_content = p.read_text(encoding='utf-8')
        if new_str not in verify_content:
            return EditResult(
                status=EditStatus.ERROR,
                message=f"Warning: File {p} was edited but verification failed. The new content may not have been written correctly.",
                path=p
            )
    except Exception as e:
        return EditResult(
            status=EditStatus.ERROR,
            message=f"Error writing {p}: {type(e).__name__}: {e}",
            path=p
        )

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    lines_affected = new_str.count("\n") + 1
    
    return EditResult(
        status=EditStatus.SUCCESS,
        message=f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", start + 1),
        path=p,
        lines_affected=lines_affected
    )


def _insert(p: Path, line_num: int, new_str: str) -> EditResult:
    """Insert new_str after the specified line number."""
    if not p.exists():
        return EditResult(
            status=EditStatus.NOT_FOUND,
            message=f"Error: {p} does not exist.",
            path=p
        )
    
    content = p.read_text().expandtabs()
    lines = content.split("\n")

    if line_num < 0 or line_num > len(lines):
        return EditResult(
            status=EditStatus.VALIDATION_ERROR,
            message=f"Error: insert_line {line_num} out of range [0, {len(lines)}]",
            path=p
        )

    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    
    try:
        p.write_text("\n".join(new_lines), encoding='utf-8')
        snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
        lines_affected = new_str.count("\n") + 1
        
        return EditResult(
            status=EditStatus.SUCCESS,
            message=f"File {p} edited. " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3)),
            path=p,
            lines_affected=lines_affected
        )
    except Exception as e:
        return EditResult(
            status=EditStatus.ERROR,
            message=f"Error inserting into {p}: {type(e).__name__}: {e}",
            path=p
        )


def _undo(p: Path) -> EditResult:
    """Undo the last edit to the file."""
    prev = _history.undo(str(p))
    if prev is None:
        return EditResult(
            status=EditStatus.VALIDATION_ERROR,
            message=f"No edit history for {p}.",
            path=p
        )
    
    try:
        p.write_text(prev, encoding='utf-8')
        return EditResult(
            status=EditStatus.SUCCESS,
            message=f"Last edit to {p} undone.",
            path=p,
            lines_affected=len(prev.splitlines())
        )
    except Exception as e:
        return EditResult(
            status=EditStatus.ERROR,
            message=f"Error undoing edit to {p}: {type(e).__name__}: {e}",
            path=p
        )
