"""
File editor tool: view, create, str_replace, insert, undo_edit.

Reimplemented from facebookresearch/HyperAgents agent/tools/edit.py.
Same commands, same validation, same history tracking.
Enhanced with file size limits, better error handling, logging, and Python syntax validation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import py_compile
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum file size to read/edit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024
# Maximum number of undo history entries per file
MAX_UNDO_HISTORY = 10
# Maximum output length for view command
MAX_VIEW_OUTPUT = 10000


def tool_info() -> dict:
    return {
        "name": "editor",
        "description": (
            "File editor for viewing, creating, and editing files. "
            "Commands: view, create, str_replace, insert, undo_edit. "
            "str_replace requires old_str to match exactly and be unique. "
            "Paths must be absolute. Max file size: 10MB."
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
                    "description": "String to replace (str_replace). Must match exactly and be unique.",
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
    """Track file edit history with size limits and optional disk persistence."""
    
    def __init__(self, persist_dir: str | None = None) -> None:
        self._history: dict[str, list[str]] = {}
        self._persist_dir = persist_dir
        self._persist_lock = threading.Lock()
        
        # Load any existing persisted history
        if persist_dir:
            self._load_persisted_history()
    
    def _get_persist_path(self, path: str) -> str | None:
        """Get the persistence file path for a given file path."""
        if not self._persist_dir:
            return None
        # Create a safe filename from the path
        safe_name = hashlib.md5(path.encode()).hexdigest() + ".hist"
        return os.path.join(self._persist_dir, safe_name)
    
    def _load_persisted_history(self) -> None:
        """Load persisted history from disk."""
        if not self._persist_dir or not os.path.exists(self._persist_dir):
            return
        
        try:
            for filename in os.listdir(self._persist_dir):
                if filename.endswith(".hist"):
                    filepath = os.path.join(self._persist_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            path = data.get('path', '')
                            history = data.get('history', [])
                            if path and history:
                                self._history[path] = history[-MAX_UNDO_HISTORY:]
                    except Exception as e:
                        logger.warning(f"Failed to load history from {filepath}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load persisted history: {e}")
    
    def _persist_history(self, path: str) -> None:
        """Persist history to disk."""
        if not self._persist_dir:
            return
        
        with self._persist_lock:
            try:
                os.makedirs(self._persist_dir, exist_ok=True)
                persist_path = self._get_persist_path(path)
                if persist_path and path in self._history:
                    data = {
                        'path': path,
                        'history': self._history[path],
                        'timestamp': time.time()
                    }
                    with open(persist_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to persist history for {path}: {e}")

    def add(self, path: str, content: str) -> None:
        """Add content to history, limiting history size."""
        if path not in self._history:
            self._history[path] = []
        self._history[path].append(content)
        # Limit history size
        if len(self._history[path]) > MAX_UNDO_HISTORY:
            self._history[path].pop(0)
        logger.debug(f"Added history for {path}, history size: {len(self._history[path])}")
        
        # Persist to disk
        self._persist_history(path)

    def undo(self, path: str) -> str | None:
        """Undo last edit."""
        if path in self._history and self._history[path]:
            content = self._history[path].pop()
            logger.info(f"Undo for {path}, remaining history: {len(self._history[path])}")
            
            # Update persisted history
            self._persist_history(path)
            
            return content
        return None
    
    def clear(self, path: str) -> None:
        """Clear history for a path."""
        if path in self._history:
            del self._history[path]
            
            # Remove persisted file if exists
            if self._persist_dir:
                persist_path = self._get_persist_path(path)
                if persist_path and os.path.exists(persist_path):
                    try:
                        os.remove(persist_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove persisted history for {path}: {e}")


# Initialize history with optional disk persistence
# Use a temp directory for persistence by default
_DEFAULT_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "agent_editor_history")
_history = _FileHistory(persist_dir=_DEFAULT_PERSIST_DIR)


def _get_file_hash(content: str) -> str:
    """Get MD5 hash of content for deduplication."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _validate_python_syntax(content: str, filepath: str = "<unknown>") -> tuple[bool, str]:
    """Validate Python syntax using py_compile.
    
    Returns:
        (is_valid, error_message)
    """
    # Only validate if it looks like Python code
    if not filepath.endswith(".py"):
        return True, ""
    
    try:
        # Create a temporary file to compile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            py_compile.compile(tmp_path, doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            # Extract the error message without the temp file path
            error_msg = str(e)
            if tmp_path in error_msg:
                error_msg = error_msg.replace(tmp_path, filepath)
            return False, f"Python syntax error: {error_msg}"
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.warning(f"Failed to validate Python syntax: {e}")
        # If validation fails, allow the edit (non-blocking)
        return True, ""


def _check_file_size(path: str | Path) -> tuple[bool, int]:
    """Check if file size is within limits. Returns (ok, size)."""
    try:
        size = os.path.getsize(path)
        return size <= MAX_FILE_SIZE, size
    except OSError:
        return True, 0  # Assume ok if we can't check


def _truncate(content: str, max_len: int = MAX_VIEW_OUTPUT) -> str:
    """Truncate content if it exceeds max_len."""
    if len(content) > max_len:
        half = max_len // 2
        return content[:half] + f"\n<response clipped: {len(content)} chars total>\n" + content[-half:]
    return content


def _format_output(content: str, path: str, init_line: int = 1) -> str:
    """Format output with line numbers."""
    content = _truncate(content).expandtabs()
    numbered = [f"{i + init_line:6}\t{line}" for i, line in enumerate(content.split("\n"))]
    return f"Here's the result of running `cat -n` on {path}:\n" + "\n".join(numbered) + "\n"


def _create_backup(path: Path) -> Path | None:
    """Create a backup of the file before editing."""
    if not path.exists() or path.is_dir():
        return None
    try:
        backup_path = Path(f"{path}.backup")
        shutil.copy2(path, backup_path)
        return backup_path
    except Exception as e:
        logger.warning(f"Failed to create backup for {path}: {e}")
        return None


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict editor operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)
    logger.info(f"Editor allowed root set to: {_ALLOWED_ROOT}")


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
                logger.warning(f"Access denied: {resolved} outside {_ALLOWED_ROOT}")
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if not file_text:
                return "Error: file_text required for create."
            return _create(p, file_text)
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
        logger.exception(f"Unexpected error in editor tool: {command} {path}")
        return f"Error: {type(e).__name__}: {e}"


def _create(p: Path, file_text: str) -> str:
    """Create a new file."""
    if p.exists():
        if p.is_dir():
            return f"Error: {p} is a directory."
        return f"Error: {p} already exists. Use str_replace to edit."
    
    # Check content size
    content_bytes = file_text.encode("utf-8")
    if len(content_bytes) > MAX_FILE_SIZE:
        return f"Error: Content too large ({len(content_bytes)} bytes, max: {MAX_FILE_SIZE})"
    
    # Validate Python syntax for .py files
    if str(p).endswith(".py"):
        is_valid, error_msg = _validate_python_syntax(file_text, str(p))
        if not is_valid:
            logger.warning(f"Python syntax validation failed for {p}: {error_msg}")
            return f"Error: {error_msg}"
    
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text, encoding="utf-8")
        _history.add(str(p), file_text)
        logger.info(f"Created file: {p} ({len(file_text)} chars)")
        return f"File created at: {p} ({len(file_text)} characters)"
    except Exception as e:
        logger.exception(f"Failed to create file: {p}")
        return f"Error: Failed to create file: {e}"


def _view(p: Path, view_range: list[int] | None) -> str:
    """View file or directory."""
    if p.is_dir():
        try:
            result = subprocess.run(
                ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
                capture_output=True, text=True, timeout=30,
            )
            return f"Files in {p}:\n{_truncate(result.stdout, 5000)}"
        except subprocess.TimeoutExpired:
            return f"Error: Directory listing timed out for {p}"
        except Exception as e:
            return f"Error listing directory: {e}"

    if not p.exists():
        return f"Error: {p} does not exist."

    # Check file size
    ok, size = _check_file_size(p)
    if not ok:
        return f"Error: File too large ({size} bytes, max: {MAX_FILE_SIZE}). Use view_range to view parts."

    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"
    except Exception as e:
        return f"Error reading file: {e}"

    if view_range:
        lines = content.split("\n")
        start, end = view_range
        if end == -1:
            end = len(lines)
        # Validate range
        if start < 1:
            return f"Error: view_range start must be >= 1"
        if start > len(lines):
            return f"Error: view_range start ({start}) > total lines ({len(lines)})"
        if end > len(lines):
            end = len(lines)
        if start > end:
            return f"Error: view_range start ({start}) > end ({end})"
        
        content = "\n".join(lines[start - 1 : end])
        return _format_output(content, str(p), start)
    
    return _format_output(content, str(p))


def _replace(p: Path, old_str: str, new_str: str) -> str:
    """Replace old_str with new_str in file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if p.is_dir():
        return f"Error: {p} is a directory."

    # Check file size
    ok, size = _check_file_size(p)
    if not ok:
        return f"Error: File too large ({size} bytes, max: {MAX_FILE_SIZE})"

    try:
        content = p.read_text(encoding="utf-8").expandtabs()
    except Exception as e:
        return f"Error reading file: {e}"
    
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context
        snippet = content[:500] + "..." if len(content) > 500 else content
        return f"Error: old_str not found in {p}. File starts with:\n{snippet}"
    if count > 1:
        # Find all occurrences
        positions = []
        start = 0
        for i in range(min(count, 5)):  # Show first 5
            pos = content.find(old_str, start)
            if pos == -1:
                break
            line_num = content[:pos].count("\n") + 1
            positions.append(f"line {line_num}")
            start = pos + 1
        
        pos_str = ", ".join(positions)
        if count > 5:
            pos_str += f" and {count - 5} more"
        return f"Error: old_str appears {count} times in {p} at: {pos_str}. Make it unique."

    # Create backup
    backup = _create_backup(p)

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str, 1)
    
    # Validate Python syntax for .py files
    if str(p).endswith(".py"):
        is_valid, error_msg = _validate_python_syntax(new_content, str(p))
        if not is_valid:
            logger.warning(f"Python syntax validation failed for {p}: {error_msg}")
            return f"Error: {error_msg}"
    
    try:
        p.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"Error writing file: {e}"

    logger.info(f"Edited file: {p} (replaced {len(old_str)} chars with {len(new_str)} chars)")

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited. " + _format_output(snippet, f"snippet of {p}", start + 1)


def _insert(p: Path, line_num: int, new_str: str) -> str:
    """Insert new_str after line_num."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if p.is_dir():
        return f"Error: {p} is a directory."

    # Check file size
    ok, size = _check_file_size(p)
    if not ok:
        return f"Error: File too large ({size} bytes, max: {MAX_FILE_SIZE})"

    try:
        content = p.read_text(encoding="utf-8").expandtabs()
    except Exception as e:
        return f"Error reading file: {e}"
    
    lines = content.split("\n")

    if line_num < 0:
        return f"Error: insert_line must be >= 0"
    if line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]"

    # Create backup
    backup = _create_backup(p)

    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    new_content = "\n".join(new_lines)
    
    # Validate Python syntax for .py files
    if str(p).endswith(".py"):
        is_valid, error_msg = _validate_python_syntax(new_content, str(p))
        if not is_valid:
            logger.warning(f"Python syntax validation failed for {p}: {error_msg}")
            return f"Error: {error_msg}"
    
    try:
        p.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"Error writing file: {e}"

    logger.info(f"Inserted into file: {p} at line {line_num}")

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
    return f"File {p} edited. " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))


def _undo(p: Path) -> str:
    """Undo last edit."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}."
    
    try:
        p.write_text(prev, encoding="utf-8")
        logger.info(f"Undo successful: {p}")
        return f"Last edit to {p} undone."
    except Exception as e:
        return f"Error during undo: {e}"
