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
    """Execute a file editor command with enhanced validation and error handling."""
    # Validate command
    valid_commands = ["view", "create", "str_replace", "insert", "undo_edit"]
    if command not in valid_commands:
        return f"Error: unknown command '{command}'. Valid commands: {', '.join(valid_commands)}"
    
    # Validate path
    if not path:
        return "Error: path is required"
    
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path. Please provide an absolute path (e.g., /home/user/file.txt)"

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            try:
                resolved = os.path.abspath(str(p))
                # Use os.path.commonpath for proper path comparison
                if not resolved.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Path '{resolved}' is outside allowed root '{_ALLOWED_ROOT}'"
            except Exception as e:
                return f"Error: path validation failed: {e}"

        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: file_text required for create command."
            if p.exists():
                return f"Error: {path} already exists. Use str_replace to edit existing files."
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(file_text, encoding='utf-8')
                _history.add(str(p), file_text)
                return f"File created successfully at: {path}"
            except PermissionError:
                return f"Error: Permission denied when creating {path}"
            except OSError as e:
                return f"Error: Failed to create file {path}: {e}"
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str required for str_replace command."
            return _replace(p, old_str, new_str or "")
        elif command == "insert":
            if insert_line is None:
                return "Error: insert_line required for insert command."
            if new_str is None:
                return "Error: new_str required for insert command."
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _view(p: Path, view_range: list[int] | None) -> str:
    """View file or directory contents with optional metadata."""
    if p.is_dir():
        try:
            result = subprocess.run(
                ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return f"Error listing directory {p}: {result.stderr}"
            
            # Add directory metadata
            file_count = len([l for l in result.stdout.strip().split('\n') if l])
            output = f"Directory: {p}\n"
            output += f"Total items: {file_count}\n"
            output += f"---\n"
            output += f"Files in {p}:\n{_truncate(result.stdout, 5000)}"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Timeout while listing directory {p} (too many files)"
        except Exception as e:
            return f"Error listing directory {p}: {e}"

    if not p.exists():
        return f"Error: {p} does not exist."

    try:
        content = p.read_text(encoding='utf-8')
        # Get file metadata
        stat = p.stat()
        size_bytes = stat.st_size
        size_str = f"{size_bytes} bytes"
        if size_bytes > 1024:
            size_str = f"{size_bytes / 1024:.1f} KB ({size_bytes} bytes)"
        if size_bytes > 1024 * 1024:
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB ({size_bytes} bytes)"
        
        line_count = len(content.split("\n"))
        
        metadata = f"File: {p}\n"
        metadata += f"Size: {size_str}\n"
        metadata += f"Lines: {line_count}\n"
        metadata += f"---\n"
    except UnicodeDecodeError:
        return f"Error: {p} contains non-text data (binary file)."
    except PermissionError:
        return f"Error: Permission denied when reading {p}"
    except Exception as e:
        return f"Error reading {p}: {e}"
    
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
        return metadata + _format_output(content, str(p), start)
    return metadata + _format_output(content, str(p))


def _replace(p: Path, old_str: str, new_str: str) -> str:
    """Replace old_str with new_str in file p."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    try:
        content = p.read_text(encoding='utf-8').expandtabs()
    except UnicodeDecodeError:
        return f"Error: {p} contains non-text data (binary file)."
    except PermissionError:
        return f"Error: Permission denied when reading {p}"
    
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context about what was searched
        content_preview = content[:500] if len(content) > 500 else content
        return f"Error: old_str not found in {p}. Content preview:\n{content_preview}"
    if count > 1:
        # Find line numbers where old_str appears
        lines = content.split("\n")
        occurrences = []
        current_pos = 0
        for i, line in enumerate(lines, 1):
            if old_str in line:
                occurrences.append(i)
            current_pos += len(line) + 1
        return f"Error: old_str appears {count} times in {p} at lines {occurrences[:5]}{'...' if len(occurrences) > 5 else ''}. Make it unique by including more context."

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
    
    try:
        p.write_text(new_content, encoding='utf-8')
    except PermissionError:
        return f"Error: Permission denied when writing {p}"
    except OSError as e:
        return f"Error: Failed to write {p}: {e}"

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", start + 1)


def _insert(p: Path, line_num: int, new_str: str) -> str:
    """Insert new_str after line_num in file p."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    try:
        content = p.read_text(encoding='utf-8').expandtabs()
    except UnicodeDecodeError:
        return f"Error: {p} contains non-text data (binary file)."
    except PermissionError:
        return f"Error: Permission denied when reading {p}"
    
    lines = content.split("\n")

    if line_num < 0 or line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]. File has {len(lines)} lines."

    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    
    try:
        p.write_text("\n".join(new_lines), encoding='utf-8')
    except PermissionError:
        return f"Error: Permission denied when writing {p}"
    except OSError as e:
        return f"Error: Failed to write {p}: {e}"

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))


def _undo(p: Path) -> str:
    """Undo the last edit to file p."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}. Cannot undo."
    
    try:
        p.write_text(prev, encoding='utf-8')
        return f"Last edit to {p} undone successfully. File restored to previous state."
    except PermissionError:
        return f"Error: Permission denied when writing {p}"
    except OSError as e:
        return f"Error: Failed to undo edit to {p}: {e}"
