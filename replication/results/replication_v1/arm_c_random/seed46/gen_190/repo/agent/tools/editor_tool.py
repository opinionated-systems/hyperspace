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


def _validate_file_content(content: str) -> str | None:
    """Validate file content for common issues. Returns error message or None if valid."""
    # Check for file size first (fastest check)
    if len(content) > 10_000_000:  # 10MB limit
        return f"File too large ({len(content)} bytes, max 10MB)"
    
    # Check for null bytes (indicates binary file)
    if '\x00' in content:
        return "File contains null bytes - appears to be binary, not text"
    
    # Check for very long lines (might indicate encoding issues)
    lines = content.split('\n')
    max_line_len = 10000
    for i, line in enumerate(lines):
        if len(line) > max_line_len:
            return f"Line {i+1} is extremely long ({len(line)} chars), may indicate encoding issues"
    
    return None


def _is_binary_file(path: Path) -> bool:
    """Check if a file is binary by reading first 8KB and checking for null bytes."""
    try:
        with open(path, 'rb') as f:
            chunk = f.read(8192)
            return b'\x00' in chunk
    except Exception:
        return False


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
    # Validate command
    valid_commands = ["view", "create", "str_replace", "insert", "undo_edit"]
    if command not in valid_commands:
        return f"Error: unknown command '{command}'. Valid commands: {valid_commands}"
    
    # Validate path
    if not path or not isinstance(path, str):
        return "Error: path must be a non-empty string"
    
    path = path.strip()
    if not path:
        return "Error: path cannot be empty or whitespace only"
    
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path. Please provide an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Path '{path}' is outside allowed root '{_ALLOWED_ROOT}'"

        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: file_text required for create command."
            if p.exists():
                return f"Error: {path} already exists. Use str_replace to edit existing files."
            
            # Validate file content for common issues
            validation_error = _validate_file_content(file_text)
            if validation_error:
                return f"Error: {validation_error}"
            
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(file_text, encoding='utf-8')
                _history.add(str(p), file_text)
                return f"File created successfully at: {path} ({len(file_text)} characters)"
            except (IOError, OSError) as e:
                return f"Error creating file: {e}"
            except Exception as e:
                return f"Error creating file: {type(e).__name__}: {e}"
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str required for str_replace command."
            if old_str == "":
                return "Error: old_str cannot be empty. Use 'insert' command to add new content."
            return _replace(p, old_str, new_str or "")
        elif command == "insert":
            if insert_line is None:
                return "Error: insert_line required for insert command."
            if new_str is None:
                return "Error: new_str required for insert command."
            if new_str == "":
                return "Error: new_str cannot be empty."
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        import traceback
        return f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _view(p: Path, view_range: list[int] | None) -> str:
    if p.is_dir():
        result = subprocess.run(
            ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
            capture_output=True, text=True,
        )
        return f"Files in {p}:\n{_truncate(result.stdout, 5000)}"

    if not p.exists():
        return f"Error: {p} does not exist."

    # Check if file is binary before trying to read as text
    if _is_binary_file(p):
        # Get file info for binary files
        size = p.stat().st_size
        return f"Error: {p} is a binary file ({size} bytes). Cannot view binary files with editor tool."

    try:
        content = p.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return f"Error: {p} contains non-UTF-8 content and cannot be displayed as text."
    except Exception as e:
        return f"Error reading file {p}: {type(e).__name__}: {e}"
    
    if view_range:
        lines = content.split("\n")
        start, end = view_range
        if end == -1:
            end = len(lines)
        # Validate range
        if start < 1:
            start = 1
        if end > len(lines):
            end = len(lines)
        if start > end:
            return f"Error: Invalid view range [{start}, {end}]. Start must be <= end."
        content = "\n".join(lines[start - 1 : end])
        return _format_output(content, str(p), start)
    return _format_output(content, str(p))


def _replace(p: Path, old_str: str, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    
    try:
        content = p.read_text().expandtabs()
    except Exception as e:
        return f"Error reading file {p}: {type(e).__name__}: {e}"
    
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context about what was searched
        old_lines = old_str.split('\n')
        content_lines = content.split('\n')
        
        # Try to find partial matches
        partial_matches = []
        first_line = old_lines[0] if old_lines else ""
        for i, line in enumerate(content_lines):
            if first_line in line:
                partial_matches.append((i+1, line[:80]))
        
        hint = ""
        if partial_matches:
            hint = f"\nHint: Found {len(partial_matches)} line(s) containing the first line of old_str:"
            for line_num, line_text in partial_matches[:3]:
                hint += f"\n  Line {line_num}: {line_text[:60]}..."
        
        return f"Error: old_str not found in {p}. The exact text was not found.{hint}"
    
    if count > 1:
        # Show where the duplicates are
        positions = []
        start = 0
        for _ in range(min(count, 5)):  # Show up to 5 positions
            pos = content.find(old_str, start)
            if pos == -1:
                break
            line_num = content[:pos].count('\n') + 1
            positions.append(line_num)
            start = pos + 1
        
        pos_str = ", ".join(f"line {p}" for p in positions)
        if count > 5:
            pos_str += f" and {count - 5} more"
        
        return f"Error: old_str appears {count} times in {p} at {pos_str}. Make it unique by including more context."

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
    
    try:
        p.write_text(new_content)
    except Exception as e:
        return f"Error writing file {p}: {type(e).__name__}: {e}"

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", start + 1)


def _insert(p: Path, line_num: int, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    
    try:
        content = p.read_text().expandtabs()
    except Exception as e:
        return f"Error reading file {p}: {type(e).__name__}: {e}"
    
    lines = content.split("\n")

    if line_num < 0 or line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]. File has {len(lines)} lines."

    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    
    try:
        p.write_text("\n".join(new_lines))
    except Exception as e:
        return f"Error writing file {p}: {type(e).__name__}: {e}"

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))


def _undo(p: Path) -> str:
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}."
    p.write_text(prev)
    return f"Last edit to {p} undone."
