"""
File editor tool: view, create, str_replace, insert, undo_edit.

Reimplemented from facebookresearch/HyperAgents agent/tools/edit.py.
Same commands, same validation, same history tracking.

Enhanced with automatic backup functionality for safer edits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "editor",
        "description": (
            "File editor for viewing, creating, and editing files. "
            "Commands: view, create, str_replace, insert, undo_edit, file_info. "
            "str_replace requires old_str to match exactly and be unique. "
            "file_info returns metadata (size, mtime, permissions). "
            "Paths must be absolute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit", "file_info"],
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

# Backup configuration
_BACKUP_DIR: str | None = None
_MAX_BACKUPS: int = 5


def set_backup_dir(backup_dir: str | None) -> None:
    """Set the directory for storing file backups. None disables backups."""
    global _BACKUP_DIR
    _BACKUP_DIR = backup_dir
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)


def _create_backup(file_path: str) -> str | None:
    """Create a backup of a file before editing. Returns backup path or None."""
    if _BACKUP_DIR is None:
        return None
    
    try:
        p = Path(file_path)
        if not p.exists() or p.is_dir():
            return None
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{p.name}.{timestamp}.bak"
        backup_path = Path(_BACKUP_DIR) / backup_name
        
        # Copy file to backup
        shutil.copy2(file_path, backup_path)
        
        # Clean up old backups for this file
        _cleanup_old_backups(p.name)
        
        return str(backup_path)
    except Exception:
        return None


def _cleanup_old_backups(base_name: str) -> None:
    """Keep only the most recent _MAX_BACKUPS backups for a file."""
    if _BACKUP_DIR is None:
        return
    
    try:
        backup_dir = Path(_BACKUP_DIR)
        backups = sorted(
            backup_dir.glob(f"{base_name}.*.bak"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old backups beyond the limit
        for old_backup in backups[_MAX_BACKUPS:]:
            old_backup.unlink(missing_ok=True)
    except Exception:
        pass


def list_backups(file_name: str | None = None) -> list[str]:
    """List available backups. If file_name is None, list all backups."""
    if _BACKUP_DIR is None:
        return []
    
    backup_dir = Path(_BACKUP_DIR)
    if file_name:
        return [str(p) for p in backup_dir.glob(f"{file_name}.*.bak")]
    else:
        return [str(p) for p in backup_dir.glob("*.bak")]


def restore_backup(backup_path: str) -> bool:
    """Restore a file from a backup."""
    try:
        p = Path(backup_path)
        if not p.exists():
            return False
        
        # Extract original filename (everything before first .timestamp.bak)
        name = p.name
        if ".bak" not in name:
            return False
        
        # Find the original file path
        original_name = name.split(".")[0]
        
        # Look for the original file in the allowed root
        if _ALLOWED_ROOT:
            for root_file in Path(_ALLOWED_ROOT).rglob("*"):
                if root_file.is_file() and root_file.name == original_name:
                    # Backup current before restoring
                    _create_backup(str(root_file))
                    shutil.copy2(backup_path, root_file)
                    return True
        
        return False
    except Exception:
        return False


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
        elif command == "file_info":
            return _file_info(p)
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

    # Create backup before editing
    backup_path = _create_backup(str(p))
    
    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
    p.write_text(new_content)

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    
    result = f"File {p} edited. "
    if backup_path:
        result += f"Backup created at: {backup_path}. "
    result += _format_output(snippet, f"snippet of {p}", start + 1)
    return result


def _insert(p: Path, line_num: int, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    content = p.read_text().expandtabs()
    lines = content.split("\n")

    if line_num < 0 or line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]"

    # Create backup before editing
    backup_path = _create_backup(str(p))
    
    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    p.write_text("\n".join(new_lines))

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
    
    result = f"File {p} edited. "
    if backup_path:
        result += f"Backup created at: {backup_path}. "
    result += _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))
    return result


def _undo(p: Path) -> str:
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}."
    p.write_text(prev)
    return f"Last edit to {p} undone."


def _file_info(p: Path) -> str:
    """Return detailed file metadata."""
    if not p.exists():
        return f"Error: {p} does not exist."

    import time
    stat = p.stat()
    size = stat.st_size
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
    mode = oct(stat.st_mode)[-3:]
    is_dir = "directory" if p.is_dir() else "file"

    info_lines = [
        f"Path: {p}",
        f"Type: {is_dir}",
        f"Size: {size} bytes",
        f"Modified: {mtime}",
        f"Permissions: {mode}",
    ]

    if p.is_file():
        try:
            line_count = len(p.read_text().split("\n"))
            info_lines.append(f"Lines: {line_count}")
        except Exception:
            pass

    return "\n".join(info_lines)
