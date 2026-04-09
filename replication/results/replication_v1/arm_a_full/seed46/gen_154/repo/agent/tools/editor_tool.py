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
            "Commands: view, create, str_replace, insert, undo_edit, search. "
            "str_replace requires old_str to match exactly and be unique. "
            "Paths must be absolute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit", "search"],
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
                    "description": "Text to search for (search command).",
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
    """Execute a file editor command with enhanced validation."""
    try:
        # Validate command first
        if not command:
            return "Error: command is required."
        
        command = str(command).strip().lower()
        
        # Validate path
        if not path:
            return "Error: path is required."
        
        # Ensure path is a string
        if not isinstance(path, str):
            path = str(path)
        
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path. Please provide an absolute path (e.g., /workspaces/project/file.py)"

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}. You tried to access: {resolved}"

        # Validate command
        valid_commands = {"view", "create", "str_replace", "insert", "undo_edit", "search"}
        if command not in valid_commands:
            return f"Error: unknown command '{command}'. Valid commands: {', '.join(sorted(valid_commands))}"

        if command == "view":
            return _view(p, view_range)
        elif command == "create":
            if file_text is None:
                return "Error: file_text required for create command."
            if p.exists():
                return f"Error: {path} already exists. Use str_replace to edit existing files."
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(file_text)
                _history.add(str(p), file_text)
                return f"File created successfully at: {path}"
            except PermissionError:
                return f"Error: Permission denied when creating {path}. Check directory permissions."
        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str required for str_replace command."
            if new_str is None:
                return "Error: new_str required for str_replace command."
            return _replace(p, old_str, new_str)
        elif command == "insert":
            if insert_line is None:
                return "Error: insert_line required for insert command."
            if new_str is None:
                return "Error: new_str required for insert command."
            return _insert(p, insert_line, new_str)
        elif command == "undo_edit":
            return _undo(p)
        elif command == "search":
            if search_term is None:
                return "Error: search_term required for search command."
            if not search_term:
                return "Error: search_term cannot be empty."
            return _search(p, search_term)
        else:
            return f"Error: unknown command {command}"
    except PermissionError as e:
        return f"Error: Permission denied - {e}. Check file/directory permissions."
    except FileNotFoundError as e:
        return f"Error: File or directory not found - {e}. Check that the path exists."
    except IsADirectoryError as e:
        return f"Error: Expected a file but found a directory - {e}."
    except NotADirectoryError as e:
        return f"Error: Expected a directory but found a file - {e}."
    except Exception as e:
        import traceback
        return f"Error: {e}\nTraceback: {traceback.format_exc()}"


def _view(p: Path, view_range: list[int] | None) -> str:
    if p.is_dir():
        try:
            result = subprocess.run(
                ["find", str(p), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return f"Error listing directory {p}: {result.stderr}"
            return f"Files in {p}:\n{_truncate(result.stdout, 5000)}"
        except subprocess.TimeoutExpired:
            return f"Error: Timeout while listing directory {p}. Directory may be too large."
        except Exception as e:
            return f"Error listing directory {p}: {e}"

    if not p.exists():
        return f"Error: {p} does not exist. Use 'create' command to create new files."
    
    if not p.is_file():
        return f"Error: {p} is not a regular file."

    try:
        content = p.read_text()
    except UnicodeDecodeError:
        return f"Error: {p} appears to be a binary file and cannot be viewed as text."
    except PermissionError:
        return f"Error: Permission denied when reading {p}. Check file permissions."
    
    if view_range:
        lines = content.split("\n")
        start, end = view_range
        if start < 1:
            return f"Error: view_range start ({start}) must be >= 1"
        if end == -1:
            end = len(lines)
        if end > len(lines):
            end = len(lines)
        if start > end:
            return f"Error: view_range start ({start}) must be <= end ({end})"
        content = "\n".join(lines[start - 1 : end])
        return _format_output(content, str(p), start)
    return _format_output(content, str(p))


def _replace(p: Path, old_str: str, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist. Use 'create' command to create new files."
    if not p.is_file():
        return f"Error: {p} is not a file. Cannot perform str_replace on directories."
    
    # Validate inputs
    if old_str is None:
        return "Error: old_str is required for str_replace."
    if new_str is None:
        return "Error: new_str is required for str_replace."
    
    # Convert to strings if needed
    if not isinstance(old_str, str):
        old_str = str(old_str)
    if not isinstance(new_str, str):
        new_str = str(new_str)
    
    content = p.read_text().expandtabs()
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs()

    count = content.count(old_str)
    if count == 0:
        # Provide helpful context about what was searched for
        snippet_preview = old_str[:100] + "..." if len(old_str) > 100 else old_str
        # Try to find similar content
        similar_lines = []
        old_lines = old_str.strip().split('\n')
        if old_lines:
            first_line = old_lines[0].strip()
            if len(first_line) > 10:
                for i, line in enumerate(content.split('\n')):
                    if first_line in line:
                        similar_lines.append((i+1, line.strip()))
        
        error_msg = f"Error: old_str not found in {p}\n"
        error_msg += f"Searched for: {repr(snippet_preview)}"
        if similar_lines:
            error_msg += f"\n\nFound {len(similar_lines)} similar line(s):"
            for line_num, line_text in similar_lines[:5]:
                error_msg += f"\n  Line {line_num}: {line_text[:80]}"
        return error_msg
    
    if count > 1:
        # Show where the duplicates are
        positions = []
        start_idx = 0
        for _ in range(count):
            idx = content.find(old_str, start_idx)
            line_num = content[:idx].count("\n") + 1
            positions.append(line_num)
            start_idx = idx + 1
        return f"Error: old_str appears {count} times in {p} at lines {positions}. Make it unique by including more context."

    _history.add(str(p), content)
    new_content = content.replace(old_str, new_str)
    
    # Write with error handling
    try:
        p.write_text(new_content)
    except PermissionError:
        return f"Error: Permission denied when writing to {p}. Check file permissions."
    except OSError as e:
        return f"Error: OS error when writing to {p}: {e}"

    # Show context around edit
    line_num = content.split(old_str)[0].count("\n")
    start = max(0, line_num - 4)
    end = line_num + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start : end + 1])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", start + 1)


def _insert(p: Path, line_num: int, new_str: str) -> str:
    if not p.exists():
        return f"Error: {p} does not exist. Use 'create' command to create new files."
    if not p.is_file():
        return f"Error: {p} is not a file. Cannot perform insert on directories."
    
    # Validate inputs
    if new_str is None:
        return "Error: new_str is required for insert."
    if not isinstance(new_str, str):
        new_str = str(new_str)
    
    # Validate line_num
    if line_num is None:
        return "Error: insert_line is required for insert."
    try:
        line_num = int(line_num)
    except (ValueError, TypeError):
        return f"Error: insert_line must be an integer, got {type(line_num).__name__}"
    
    content = p.read_text().expandtabs()
    lines = content.split("\n")

    if line_num < 0 or line_num > len(lines):
        return f"Error: insert_line {line_num} out of range [0, {len(lines)}]. File has {len(lines)} lines."

    _history.add(str(p), content)
    new_lines = lines[:line_num] + new_str.expandtabs().split("\n") + lines[line_num:]
    
    # Write with error handling
    try:
        p.write_text("\n".join(new_lines))
    except PermissionError:
        return f"Error: Permission denied when writing to {p}. Check file permissions."
    except OSError as e:
        return f"Error: OS error when writing to {p}: {e}"

    snippet = "\n".join(new_lines[max(0, line_num - 4) : line_num + 4 + new_str.count("\n")])
    return f"File {p} edited successfully. " + _format_output(snippet, f"snippet of {p}", max(1, line_num - 3))


def _undo(p: Path) -> str:
    if not p.exists():
        return f"Error: {p} does not exist. Cannot undo edits on non-existent file."
    
    if not p.is_file():
        return f"Error: {p} is not a file. Cannot undo edits on directories."
    
    prev = _history.undo(str(p))
    if prev is None:
        return f"No edit history for {p}. Only str_replace and insert operations are tracked for undo."
    
    try:
        p.write_text(prev)
        return f"Last edit to {p} undone successfully. File restored to previous state."
    except PermissionError:
        return f"Error: Permission denied when undoing edit to {p}. Check file permissions."
    except OSError as e:
        return f"Error: OS error when undoing edit to {p}: {e}"


def _search(p: Path, search_term: str) -> str:
    """Search for a term in a file or directory.
    
    If path is a file, searches within that file.
    If path is a directory, recursively searches all files.
    Returns line numbers and context for each match.
    """
    import re
    
    if not p.exists():
        return f"Error: {p} does not exist."
    
    # Validate search_term
    if search_term is None:
        return "Error: search_term is required for search."
    
    if not isinstance(search_term, str):
        search_term = str(search_term)
    
    search_term = search_term.strip()
    if not search_term:
        return "Error: search_term cannot be empty."
    
    # Limit search term length
    MAX_SEARCH_TERM_LENGTH = 1000
    if len(search_term) > MAX_SEARCH_TERM_LENGTH:
        return f"Error: search_term too long ({len(search_term)} chars). Maximum: {MAX_SEARCH_TERM_LENGTH}"
    
    matches = []
    files_searched = 0
    max_files = 1000  # Limit to prevent timeout on huge directories
    max_matches = 100  # Limit total matches
    
    if p.is_file():
        files_to_search = [p]
    else:
        # Find all files recursively
        files_to_search = []
        for root, dirs, files in os.walk(p):
            # Skip hidden directories and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules', '.git', 'venv', '.venv', '.tox', '.pytest_cache', 'dist', 'build'}]
            for f in files:
                if not f.startswith('.') and not f.endswith(('.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.dat', '.db')):
                    files_to_search.append(Path(root) / f)
                    if len(files_to_search) >= max_files:
                        break
            if len(files_to_search) >= max_files:
                break
    
    for file_path in files_to_search:
        files_searched += 1
        try:
            # Skip binary files and very large files
            stat = file_path.stat()
            if stat.st_size > 1024 * 1024:  # Skip files > 1MB
                continue
            
            # Try to read as text
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            except (UnicodeDecodeError, OSError):
                # Skip files that can't be read as text
                continue
            
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                if search_term in line:
                    # Show context: 2 lines before and after
                    start = max(0, i - 3)
                    end = min(len(lines), i + 2)
                    context = '\n'.join(
                        f"{j+1:4}: {lines[j]}" 
                        for j in range(start, end)
                    )
                    matches.append(f"\n{file_path}:{i}\n{context}")
                    
                    # Limit matches per file to prevent huge output
                    if len(matches) >= max_matches:
                        break
        except (PermissionError, OSError):
            # Skip files that can't be read
            continue
        except Exception:
            # Skip files that cause other errors
            continue
        
        if len(matches) >= max_matches:
            break
    
    if not matches:
        return f"No matches found for '{search_term}' in {p} (searched {files_searched} files)"
    
    result = f"Found {len(matches)} match(es) for '{search_term}' in {p} (searched {files_searched} files):"
    result += ''.join(matches[:20])  # Limit to first 20 matches
    if len(matches) > 20:
        result += f"\n... and {len(matches) - 20} more matches"
    if len(files_to_search) >= max_files:
        result += f"\n... (search limited to {max_files} files)"
    return result
