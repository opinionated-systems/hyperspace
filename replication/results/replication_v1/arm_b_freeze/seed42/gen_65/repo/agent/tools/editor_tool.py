"""
File editor tool: view, create, str_replace, str_replace_all, insert, undo_edit, search.

Reimplemented from facebookresearch/HyperAgents agent/tools/edit.py.
Same commands, same validation, same history tracking.
Added str_replace_all for bulk replacements.
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
            "Commands: view, create, str_replace, str_replace_all, insert, undo_edit, search. "
            "str_replace requires old_str to match exactly and be unique. "
            "str_replace_all replaces all occurrences. "
            "Paths must be absolute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "str_replace_all", "insert", "undo_edit", "search"],
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
                "search_term": {
                    "type": "string",
                    "description": "Term to search for (search command).",
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
    search_term: str | None = None,
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
        elif command == "str_replace_all":
            if old_str is None:
                return "Error: old_str required for str_replace_all."
            return _replace_all(p, old_str, new_str or "")
        elif command == "insert":
            if insert_line is None or new_str is None:
                return "Error: insert_line and new_str required for insert."
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
        elif command == "search":
            if search_term is None:
                return "Error: search_term required for search."
            return _search(p, search_term)
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


def _replace_all(p: Path, old_str: str, new_str: str) -> str:
    """Replace all occurrences of old_str with new_str in the file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    content = p.read_text().expandtabs()
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        return f"Error: old_str not found in {p}"

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
    p.write_text(new_content)

    # Show first occurrence context
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 2)
    end = line_num + 2 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited. Replaced {count} occurrence(s). " + _format_output(snippet, f"snippet of {p}", start + 1)


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


def _search(p: Path, search_term: str) -> str:
    """Search for a term in a file or directory.
    
    For directories: search all files recursively.
    For files: search within the file content.
    """
    results = []
    
    if p.is_dir():
        # Search recursively in directory
        for root, _, files in os.walk(p):
            for filename in files:
                if filename.startswith('.') or filename.endswith('.pyc'):
                    continue
                filepath = Path(root) / filename
                try:
                    content = filepath.read_text(errors='ignore')
                    if search_term in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if search_term in line:
                                results.append(f"{filepath}:{i}: {line.strip()}")
                except Exception:
                    pass
    elif p.is_file():
        # Search in single file
        try:
            content = p.read_text()
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if search_term in line:
                    results.append(f"{p}:{i}: {line.strip()}")
        except Exception as e:
            return f"Error reading {p}: {e}"
    else:
        return f"Error: {p} does not exist."
    
    if not results:
        return f"No matches found for '{search_term}' in {p}"
    
    # Limit results to prevent overwhelming output
    max_results = 50
    if len(results) > max_results:
        truncated = results[:max_results]
        return f"Found {len(results)} matches for '{search_term}' in {p} (showing first {max_results}):\n" + "\n".join(truncated) + f"\n... and {len(results) - max_results} more matches"
    
    return f"Found {len(results)} matches for '{search_term}' in {p}:\n" + "\n".join(results)
