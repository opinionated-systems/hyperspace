"""
Search tool: find files and search content within files.

Provides grep-like functionality and file discovery capabilities.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files or content within files. "
            "Commands: find_files (glob pattern), grep (search content), "
            "find_in_files (search text in file contents)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for find_files, regex for grep/find_in_files.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '*.py').",
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True, ""
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n... [output truncated] ...\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        is_valid, error = _check_path(path)
        if not is_valid:
            return error
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        if command == "find_files":
            return _find_files(p, pattern)
        elif command == "grep":
            return _grep(p, pattern, file_pattern)
        elif command == "find_in_files":
            return _find_in_files(p, pattern, file_pattern)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(base_path: Path, pattern: str) -> str:
    """Find files matching a glob pattern."""
    try:
        matches = list(base_path.rglob(pattern))
        if not matches:
            return f"No files found matching '{pattern}' in {base_path}"
        
        # Limit output
        files = [str(m.relative_to(base_path)) for m in matches if m.is_file()]
        if len(files) > 100:
            files = files[:50] + [f"... ({len(files) - 50} more files) ..."] + files[-50:]
        
        return f"Found {len(matches)} matches:\n" + "\n".join(files)
    except Exception as e:
        return f"Error finding files: {e}"


def _grep(base_path: Path, pattern: str, file_pattern: str | None) -> str:
    """Search for pattern in file contents using grep."""
    try:
        cmd = ["grep", "-r", "-n", "-I", "--color=never", pattern]
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        cmd.append(str(base_path))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            return _truncate(result.stdout, 8000)
        elif result.returncode == 1:
            return f"No matches found for '{pattern}'"
        else:
            return f"grep error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: grep timed out (search may be too broad)"
    except FileNotFoundError:
        return "Error: grep command not found"
    except Exception as e:
        return f"Error running grep: {e}"


def _find_in_files(base_path: Path, pattern: str, file_pattern: str | None) -> str:
    """Find files containing the pattern (case-insensitive)."""
    try:
        cmd = ["grep", "-r", "-l", "-i", "-I", "--color=never", pattern]
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        cmd.append(str(base_path))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            files = result.stdout.strip().split("\n")
            if len(files) > 50:
                files = files[:25] + [f"... ({len(files) - 25} more files) ..."]
            return f"Files containing '{pattern}':\n" + "\n".join(files)
        elif result.returncode == 1:
            return f"No files found containing '{pattern}'"
        else:
            return f"grep error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: search timed out (pattern may be too broad)"
    except FileNotFoundError:
        return "Error: grep command not found"
    except Exception as e:
        return f"Error searching files: {e}"
