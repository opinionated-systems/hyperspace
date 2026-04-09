"""
File editor tool: view, create, str_replace, insert, undo_edit.

Reimplemented from facebookresearch/HyperAgents agent/tools/edit.py.
Same commands, same validation, same history tracking.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from agent.config import DEFAULT_AGENT_CONFIG


def tool_info() -> dict:
    return {
        "name": "editor",
        "description": (
            "File editor for viewing, creating, and editing files. "
            "Commands: view, view_line, create, str_replace, insert, undo_edit. "
            "str_replace requires old_str to match exactly and be unique. "
            "Paths must be absolute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "view_line", "create", "str_replace", "insert", "undo_edit"],
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
                "line_number": {
                    "type": "integer",
                    "description": "Line number to view (view_line command).",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines around target line (view_line, default: 3).",
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


def _truncate(content: str, max_len: int = DEFAULT_AGENT_CONFIG.max_file_size) -> str:
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
    line_number: int | None = None,
    context_lines: int = 3,
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
        elif command == "view_line":
            if line_number is None:
                return "Error: line_number required for view_line."
            return _view_line(p, line_number, context_lines)
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


def _view_line(p: Path, line_number: int, context_lines: int = 3) -> str:
    """View a specific line with surrounding context.
    
    Args:
        p: Path to the file
        line_number: The line number to view (1-indexed)
        context_lines: Number of lines to show before and after
        
    Returns:
        Formatted output with the line and context
    """
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if p.is_dir():
        return f"Error: {p} is a directory, not a file."
    
    content = p.read_text()
    lines = content.split("\n")
    total_lines = len(lines)
    
    if line_number < 1 or line_number > total_lines:
        return f"Error: Line {line_number} is out of range. File has {total_lines} lines (valid range: 1-{total_lines})."
    
    # Calculate context range
    start = max(1, line_number - context_lines)
    end = min(total_lines, line_number + context_lines)
    
    # Extract the lines
    context = "\n".join(lines[start - 1 : end])
    
    # Format with highlighting of the target line
    formatted_lines = []
    for i in range(start, end + 1):
        line_content = lines[i - 1]
        prefix = ">>> " if i == line_number else "    "
        formatted_lines.append(f"{prefix}{i:6d}\t{line_content}")
    
    header = f"Line {line_number} of {total_lines} in {p}:"
    return f"{header}\n" + "\n".join(formatted_lines) + "\n"


def _replace(p: Path, old_str: str, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    content = p.read_text().expandtabs()
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context about what was found
        # Try to find similar strings
        old_lines = old_str.strip().split("\n")
        if old_lines:
            first_line = old_lines[0].strip()
            if first_line:
                # Search for the first line in the file
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if first_line in line:
                        context_start = max(0, i - 3)
                        context_end = min(len(lines), i + 4)
                        context = "\n".join(f"{j+1:4d}: {lines[j]}" for j in range(context_start, context_end))
                        return f"Error: old_str not found in {p}. Did you mean around line {i+1}?\nSimilar context found:\n{context}"
        return f"Error: old_str not found in {p}"
    if count > 1:
        # Show where the duplicates are
        lines = content.split("\n")
        positions = []
        search_start = 0
        for _ in range(min(count, 5)):  # Show up to 5 positions
            idx = content.find(old_str, search_start)
            if idx == -1:
                break
            line_num = content[:idx].count("\n") + 1
            positions.append(str(line_num))
            search_start = idx + 1
        pos_str = ", ".join(positions)
        if count > 5:
            pos_str += f" and {count - 5} more"
        return f"Error: old_str appears {count} times in {p} at lines: {pos_str}. Make it unique by including more context."

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
