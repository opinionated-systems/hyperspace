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
            "Commands: view, create, str_replace, insert, undo_edit, get_history. "
            "str_replace requires old_str to match exactly and be unique. "
            "Paths must be absolute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit", "get_history"],
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


class _FileBackup:
    """Manages file backups before edits for recovery purposes."""
    
    def __init__(self, max_backups: int = 5) -> None:
        self._backups: dict[str, list[tuple[str, float]]] = {}
        self._max_backups = max_backups
    
    def backup(self, path: str, content: str) -> None:
        """Store a backup of file content with timestamp."""
        import time
        self._backups.setdefault(path, [])
        self._backups[path].append((content, time.time()))
        
        # Limit number of backups per file
        if len(self._backups[path]) > self._max_backups:
            self._backups[path].pop(0)
    
    def get_backup(self, path: str, index: int = -1) -> str | None:
        """Get a specific backup (default: most recent)."""
        if path not in self._backups or not self._backups[path]:
            return None
        try:
            return self._backups[path][index][0]
        except IndexError:
            return None
    
    def list_backups(self, path: str) -> list[tuple[int, float]]:
        """List available backups with their indices and timestamps."""
        if path not in self._backups:
            return []
        return [(i, ts) for i, (_, ts) in enumerate(self._backups[path])]
    
    def clear_backups(self, path: str) -> None:
        """Clear all backups for a file."""
        if path in self._backups:
            del self._backups[path]


class _FileModificationTracker:
    """Tracks file modifications with timestamps and edit metadata."""
    
    def __init__(self, max_entries: int = 100) -> None:
        self._modifications: dict[str, list[dict]] = {}
        self._max_entries = max_entries
        self._global_edit_count = 0
    
    def record_modification(
        self, 
        path: str, 
        operation: str, 
        old_content: str | None = None,
        new_content: str | None = None,
        details: dict | None = None
    ) -> dict:
        """Record a file modification with metadata.
        
        Args:
            path: File path
            operation: Type of operation (create, str_replace, insert, undo)
            old_content: Previous content (for tracking changes)
            new_content: New content after modification
            details: Additional operation-specific details
        
        Returns:
            The recorded modification entry
        """
        import time
        from datetime import datetime
        
        self._global_edit_count += 1
        
        # Calculate change metrics
        old_lines = len(old_content.split('\n')) if old_content else 0
        new_lines = len(new_content.split('\n')) if new_content else 0
        line_diff = new_lines - old_lines
        
        entry = {
            'edit_id': self._global_edit_count,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'operation': operation,
            'path': path,
            'old_line_count': old_lines,
            'new_line_count': new_lines,
            'line_diff': line_diff,
            'details': details or {},
        }
        
        self._modifications.setdefault(path, [])
        self._modifications[path].append(entry)
        
        # Limit entries per file
        if len(self._modifications[path]) > self._max_entries:
            self._modifications[path].pop(0)
        
        return entry
    
    def get_modifications(self, path: str, limit: int = 10) -> list[dict]:
        """Get recent modifications for a file."""
        if path not in self._modifications:
            return []
        return self._modifications[path][-limit:]
    
    def get_last_modification(self, path: str) -> dict | None:
        """Get the most recent modification for a file."""
        if path not in self._modifications or not self._modifications[path]:
            return None
        return self._modifications[path][-1]
    
    def get_all_paths(self) -> list[str]:
        """Get all paths that have been modified."""
        return list(self._modifications.keys())
    
    def get_stats(self) -> dict:
        """Get modification statistics."""
        total_edits = sum(len(mods) for mods in self._modifications.values())
        return {
            'total_edits': total_edits,
            'files_modified': len(self._modifications),
            'global_edit_count': self._global_edit_count,
        }
    
    def clear_history(self, path: str | None = None) -> None:
        """Clear modification history for a specific file or all files."""
        if path is None:
            self._modifications.clear()
            self._global_edit_count = 0
        elif path in self._modifications:
            del self._modifications[path]


_history = _FileHistory()
_backup = _FileBackup()
_modification_tracker = _FileModificationTracker()


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
    
    Args:
        command: The command to run (view, create, str_replace, insert, undo_edit)
        path: Absolute path to file or directory
        file_text: Content for create command
        view_range: Line range [start, end] for view command
        old_str: String to replace (str_replace)
        new_str: Replacement string (str_replace/insert)
        insert_line: Line number to insert after (insert)
    
    Returns:
        Result message or error
    """
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
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if command == "view":
            # Validate view_range if provided
            if view_range is not None:
                if not isinstance(view_range, list) or len(view_range) != 2:
                    return "Error: view_range must be a list of two integers [start, end]"
                if not all(isinstance(x, int) for x in view_range):
                    return "Error: view_range values must be integers"
                if view_range[0] < 1:
                    return "Error: view_range start must be >= 1"
                if view_range[1] != -1 and view_range[1] < view_range[0]:
                    return "Error: view_range end must be >= start (or -1 for end of file)"
            return _view(p, view_range)
            
        elif command == "create":
            if file_text is None:
                return "Error: file_text required for create."
            if p.exists():
                return f"Error: {path} already exists. Use str_replace to edit."
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(file_text, encoding='utf-8')
                _history.add(str(p), file_text)
                # Track the creation
                _modification_tracker.record_modification(
                    path=str(p),
                    operation="create",
                    old_content=None,
                    new_content=file_text,
                    details={'size_bytes': len(file_text.encode('utf-8'))}
                )
                return f"File created at: {path}"
            except OSError as e:
                return f"Error creating file: {e}"
            
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str required for str_replace."
            if not isinstance(old_str, str):
                return "Error: old_str must be a string"
            return _replace(p, old_str, new_str or "")
            
        elif command == "insert":
            if insert_line is None:
                return "Error: insert_line required for insert."
            if new_str is None:
                return "Error: new_str required for insert."
            if not isinstance(insert_line, int):
                return "Error: insert_line must be an integer"
            if not isinstance(new_str, str):
                return "Error: new_str must be a string"
            return _insert(p, insert_line, new_str)
            
        elif command == "undo_edit":
            return _undo(p)
        
        elif command == "get_history":
            return _get_history(p)
            
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _view(p: Path, view_range: list[int] | None) -> str:
    if p.is_dir():
        result = subprocess.run(
            ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
            capture_output=True, text=True,
        )
        return f"Files in {p}:\n{_truncate(result.stdout, 5000)}"

    if not p.exists():
        return f"Error: {p} does not exist."

    try:
        content = p.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            content = p.read_text(encoding='latin-1')
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"
    except Exception as e:
        return f"Error reading file: {type(e).__name__}: {e}"
    
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
        # Provide helpful suggestions
        suggestions = []
        
        # Check for whitespace issues
        stripped_old = old_str.strip()
        if stripped_old and content.count(stripped_old) > 0:
            suggestions.append(f"Found '{stripped_old}' without exact whitespace. Check leading/trailing spaces or tabs.")
        
        # Check for case differences
        lower_old = old_str.lower()
        if lower_old != old_str:
            lower_count = content.lower().count(lower_old)
            if lower_count > 0:
                suggestions.append(f"Found case-insensitive match. Check capitalization.")
        
        # Check for partial matches (first 50 chars)
        if len(old_str) > 10:
            partial = old_str[:50]
            if partial in content:
                suggestions.append(f"Found partial match (first 50 chars). The full string may have subtle differences.")
        
        # Suggest viewing the file
        suggestions.append(f"Use editor with command='view' and path='{p}' to see the current file content.")
        
        error_msg = f"Error: old_str not found in {p}"
        if suggestions:
            error_msg += "\n\nSuggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
        return error_msg
    
    if count > 1:
        # Find line numbers of occurrences
        lines = content.split("\n")
        occurrences = []
        current_pos = 0
        for i, line in enumerate(lines, 1):
            line_start = current_pos
            line_end = current_pos + len(line)
            line_content = content[line_start:line_end]
            
            # Check if old_str starts in this line
            if old_str in line_content:
                occurrences.append(i)
            current_pos = line_end + 1  # +1 for newline
        
        error_msg = f"Error: old_str appears {count} times in {p}. Make it unique."
        if occurrences:
            error_msg += f"\n\nOccurrences found at lines: {', '.join(map(str, occurrences[:5]))}"
            if len(occurrences) > 5:
                error_msg += f" (and {len(occurrences) - 5} more)"
        error_msg += "\n\nSuggestion: Include more context in old_str to make it unique."
        return error_msg

    # Backup before edit
    _backup.backup(str(p), content)
    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
    p.write_text(new_content, encoding='utf-8')
    
    # Track the modification
    line_num = content.split(old_str)[0].count("\n")
    _modification_tracker.record_modification(
        path=str(p),
        operation="str_replace",
        old_content=content,
        new_content=new_content,
        details={
            'old_str_preview': old_str[:50] + '...' if len(old_str) > 50 else old_str,
            'new_str_preview': new_str[:50] + '...' if len(new_str) > 50 else new_str,
            'line_changed': line_num + 1,
        }
    )

    # Show context around edit
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited. " + _format_output(snippet, f"snippet of {p}", start + 1)


def _insert(p: Path, line_num: int, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist."
    content = p.read_text(encoding='utf-8').expandtabs()
    lines = content.split("\n")

    if line_num < 0 or line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]"

    # Backup before edit
    _backup.backup(str(p), content)
    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    new_content = "\n".join(new_lines)
    p.write_text(new_content, encoding='utf-8')
    
    # Track the modification
    _modification_tracker.record_modification(
        path=str(p),
        operation="insert",
        old_content=content,
        new_content=new_content,
        details={
            'insert_line': line_num,
            'lines_inserted': len(new_str.split('\n')),
            'new_str_preview': new_str[:50] + '...' if len(new_str) > 50 else new_str,
        }
    )

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
    return f"File {p} edited. " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))


def _undo(p: Path) -> str:
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}."
    
    # Get current content before undo
    try:
        current_content = p.read_text(encoding='utf-8')
    except:
        current_content = None
    
    p.write_text(prev, encoding='utf-8')
    
    # Track the undo operation
    _modification_tracker.record_modification(
        path=str(p),
        operation="undo",
        old_content=current_content,
        new_content=prev,
        details={'restored_from_history': True}
    )
    
    return f"Last edit to {p} undone."


def _get_history(p: Path) -> str:
    """Get modification history for a file."""
    path_str = str(p)
    
    # Get file-specific modifications
    modifications = _modification_tracker.get_modifications(path_str, limit=20)
    
    if not modifications:
        return f"No modification history for {p}."
    
    lines = [f"Modification History for: {p}", "=" * 60, ""]
    
    for mod in reversed(modifications):  # Most recent first
        lines.append(f"Edit #{mod['edit_id']} - {mod['operation']}")
        lines.append(f"  Time: {mod['datetime']}")
        lines.append(f"  Lines: {mod['old_line_count']} → {mod['new_line_count']} (Δ{mod['line_diff']:+d})")
        
        if mod['details']:
            lines.append("  Details:")
            for key, value in mod['details'].items():
                if isinstance(value, str) and len(value) > 70:
                    value = value[:67] + "..."
                lines.append(f"    - {key}: {value}")
        
        lines.append("")
    
    # Add summary stats
    stats = _modification_tracker.get_stats()
    lines.append("-" * 60)
    lines.append(f"Total edits across all files: {stats['total_edits']}")
    lines.append(f"Files modified: {stats['files_modified']}")
    
    return "\n".join(lines)


def get_modification_stats() -> dict:
    """Get global modification statistics."""
    return _modification_tracker.get_stats()


def get_file_modifications(path: str, limit: int = 10) -> list[dict]:
    """Get modification history for a specific file."""
    return _modification_tracker.get_modifications(path, limit)
