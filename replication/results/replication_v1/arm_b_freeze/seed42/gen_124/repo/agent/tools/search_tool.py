"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality and file finding capabilities.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content within the codebase. "
            "Provides grep-like functionality and file finding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find", "file_regex"],
                    "description": "Search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern or regex.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py'). Optional.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default 50.",
                },
            },
            "required": ["command", "pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    command: str,
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)
        logger.debug("Search command: %s pattern: %s in path: %s", command, pattern, path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {path} does not exist."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                logger.warning("Access denied: %s outside allowed root %s", resolved, _ALLOWED_ROOT)
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if command == "grep":
            return _grep(p, pattern, file_pattern, max_results)
        elif command == "find":
            return _find(p, pattern, max_results)
        elif command == "file_regex":
            return _file_regex(p, pattern, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        logger.error("Search error: %s", e)
        return f"Error: {e}"


def _grep(directory: Path, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Search for pattern in file contents using grep."""
    if not directory.is_dir():
        return f"Error: {directory} is not a directory."

    cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", pattern, str(directory)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        lines = [line for line in lines if line]
        
        if not lines:
            return f"No matches found for '{pattern}' in {directory}"
        
        total = len(lines)
        if total > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total - max_results} more results truncated) ..."
        else:
            truncated_msg = ""
        
        output = f"Found {total} matches for '{pattern}':\n" + "\n".join(lines) + truncated_msg
        return output
    except subprocess.TimeoutExpired:
        return f"Error: grep timed out after 30 seconds"
    except Exception as e:
        return f"Error running grep: {e}"


def _find(directory: Path, pattern: str, max_results: int) -> str:
    """Find files by name pattern."""
    if not directory.is_dir():
        return f"Error: {directory} is not a directory."

    try:
        # Use find command for efficiency
        cmd = ["find", str(directory), "-type", "f", "-name", pattern]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        files = [f for f in result.stdout.strip().split("\n") if f]
        
        if not files:
            return f"No files matching '{pattern}' found in {directory}"
        
        total = len(files)
        if total > max_results:
            files = files[:max_results]
            truncated_msg = f"\n... ({total - max_results} more files truncated) ..."
        else:
            truncated_msg = ""
        
        output = f"Found {total} files matching '{pattern}':\n" + "\n".join(files) + truncated_msg
        return output
    except subprocess.TimeoutExpired:
        return f"Error: find timed out after 30 seconds"
    except Exception as e:
        return f"Error running find: {e}"


def _file_regex(directory: Path, pattern: str, max_results: int) -> str:
    """Find files by regex pattern in their content."""
    if not directory.is_dir():
        return f"Error: {directory} is not a directory."

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    matches = []
    try:
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            
            for file in files:
                if file.startswith("."):
                    continue
                    
                filepath = Path(root) / file
                try:
                    # Skip binary files and large files
                    if filepath.stat().st_size > 1024 * 1024:  # 1MB
                        continue
                        
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if regex.search(content):
                            matches.append(str(filepath))
                            if len(matches) >= max_results:
                                break
                except (IOError, OSError, UnicodeDecodeError):
                    continue
            
            if len(matches) >= max_results:
                break
        
        if not matches:
            return f"No files matching regex '{pattern}' found in {directory}"
        
        output = f"Found {len(matches)} files matching regex '{pattern}':\n" + "\n".join(matches)
        return output
    except Exception as e:
        return f"Error searching files: {e}"
