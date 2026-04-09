"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content. "
            "Commands: find_files (glob patterns), grep (content search), "
            "find_in_files (search text in files). "
            "Useful for locating code patterns and references."
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
                    "description": "Search pattern (glob for find_files, regex for grep).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to limit search (e.g., '*.py').",
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


def _check_path_allowed(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def _truncate_output(output: str, max_lines: int = 100, max_chars: int = 10000) -> str:
    """Truncate output to reasonable size."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"... ({len(lines) - max_lines} more lines truncated)")
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars // 2] + "\n... (truncated) ...\n" + result[-max_chars // 2:]
    return result


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
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        if not _check_path_allowed(str(p)):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

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


def _find_files(base_path: Path, glob_pattern: str) -> str:
    """Find files matching glob pattern."""
    try:
        matches = list(base_path.rglob(glob_pattern))
        # Filter to allowed root
        if _ALLOWED_ROOT:
            matches = [m for m in matches if str(m).startswith(_ALLOWED_ROOT)]
        
        if not matches:
            return f"No files found matching '{glob_pattern}' in {base_path}"
        
        # Sort and format
        matches_str = "\n".join(str(m) for m in sorted(matches))
        return _truncate_output(f"Found {len(matches)} files:\n{matches_str}")
    except Exception as e:
        return f"Error in find_files: {e}"


def _grep(base_path: Path, regex_pattern: str, file_pattern: str | None = None) -> str:
    """Search for regex pattern in files using grep."""
    try:
        cmd = ["grep", "-r", "-n", "-I", "--include"]
        if file_pattern:
            cmd.append(file_pattern)
        else:
            cmd.append("*")
        cmd.extend([regex_pattern, str(base_path)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            return _truncate_output(f"Matches found:\n{result.stdout}")
        elif result.returncode == 1:
            return f"No matches found for '{regex_pattern}'"
        else:
            return f"grep error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: grep timed out after 30s"
    except Exception as e:
        return f"Error in grep: {e}"


def _find_in_files(base_path: Path, text_pattern: str, file_pattern: str | None = None) -> str:
    """Simple text search in files (case-insensitive)."""
    try:
        matches = []
        search_files = base_path.rglob(file_pattern or "*")
        
        for f in search_files:
            if not f.is_file():
                continue
            if _ALLOWED_ROOT and not str(f).startswith(_ALLOWED_ROOT):
                continue
            # Skip binary files
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if text_pattern.lower() in content.lower():
                    # Find line numbers
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        if text_pattern.lower() in line.lower():
                            matches.append(f"{f}:{i}:{line.strip()}")
                            if len(matches) >= 100:
                                break
                    if len(matches) >= 100:
                        break
            except Exception:
                continue
        
        if not matches:
            return f"No files contain '{text_pattern}'"
        
        return _truncate_output(f"Found in {len(matches)} locations:\n" + "\n".join(matches))
    except Exception as e:
        return f"Error in find_in_files: {e}"
