"""
Search tool: find files and search content within files.

Provides grep-like functionality for searching file contents and
find-like functionality for locating files by name pattern.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterator

from agent.config import DEFAULT_AGENT_CONFIG


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content. "
            "Commands: grep (search file contents), find (search by filename), "
            "rg (ripgrep if available for faster searching). "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find", "rg"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for grep/rg, glob for find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern filter (e.g., '*.py' for grep/rg).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
            },
            "required": ["command", "pattern", "path"],
        },
    }


_MAX_RESULTS = 50
_MAX_OUTPUT_SIZE = DEFAULT_AGENT_CONFIG.max_file_size


def _truncate_output(output: str, max_len: int = _MAX_OUTPUT_SIZE) -> str:
    """Truncate output if it exceeds max length."""
    if len(output) > max_len:
        return output[: max_len // 2] + "\n<output clipped>\n" + output[-max_len // 2 :]
    return output


def _is_allowed_path(path: str, allowed_root: str | None) -> bool:
    """Check if path is within allowed root."""
    if allowed_root is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(allowed_root)


def _grep(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    max_results: int = _MAX_RESULTS,
    allowed_root: str | None = None,
) -> str:
    """Search file contents using Python's regex (fallback when ripgrep unavailable)."""
    if not _is_allowed_path(path, allowed_root):
        return f"Error: access denied. Search restricted to {allowed_root}"
    
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist."
    
    results: list[str] = []
    count = 0
    
    try:
        regex = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"
    
    # Determine which files to search
    if p.is_file():
        files = [p]
    else:
        if file_pattern:
            files = list(p.rglob(file_pattern))
        else:
            files = list(p.rglob("*"))
        files = [f for f in files if f.is_file() and not f.name.startswith(".")]
    
    for file_path in files:
        if count >= max_results:
            break
        
        # Skip binary files and very large files
        try:
            if file_path.stat().st_size > 10_000_000:  # Skip files > 10MB
                continue
            
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            
            for match in regex.finditer(content):
                if count >= max_results:
                    break
                
                # Find line number
                line_num = content[:match.start()].count("\n") + 1
                # Extract the line
                lines = content.split("\n")
                if 0 <= line_num - 1 < len(lines):
                    line_content = lines[line_num - 1].strip()
                    # Truncate very long lines
                    if len(line_content) > 200:
                        line_content = line_content[:200] + "..."
                    results.append(f"{file_path}:{line_num}:{line_content}")
                    count += 1
                    
        except (IOError, OSError, UnicodeDecodeError):
            continue
    
    if not results:
        return f"No matches found for pattern '{pattern}'"
    
    output = "\n".join(results)
    if count >= max_results:
        output += f"\n<results truncated at {max_results}>"
    
    return _truncate_output(output)


def _find(
    pattern: str,
    path: str,
    max_results: int = _MAX_RESULTS,
    allowed_root: str | None = None,
) -> str:
    """Find files by name pattern (glob-style)."""
    if not _is_allowed_path(path, allowed_root):
        return f"Error: access denied. Search restricted to {allowed_root}"
    
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist."
    
    if not p.is_dir():
        return f"Error: {path} is not a directory."
    
    results: list[str] = []
    count = 0
    
    # Convert glob pattern to regex
    regex_pattern = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
    try:
        regex = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: invalid pattern: {e}"
    
    for file_path in p.rglob("*"):
        if count >= max_results:
            break
        
        if regex.search(file_path.name):
            results.append(str(file_path))
            count += 1
    
    if not results:
        return f"No files matching '{pattern}' found in {path}"
    
    output = "\n".join(sorted(results))
    if count >= max_results:
        output += f"\n<results truncated at {max_results}>"
    
    return _truncate_output(output)


def _ripgrep(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    max_results: int = _MAX_RESULTS,
    allowed_root: str | None = None,
) -> str:
    """Search using ripgrep if available, fallback to Python grep."""
    if not _is_allowed_path(path, allowed_root):
        return f"Error: access denied. Search restricted to {allowed_root}"
    
    # Check if ripgrep is available
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to Python implementation
        return _grep(pattern, path, file_pattern, max_results, allowed_root)
    
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--max-count", str(max_results),
        "--max-filesize", "10M",
        "--hidden" if pattern.startswith(".") else "--no-hidden",
        pattern,
        path,
    ]
    
    if file_pattern:
        cmd.extend(["-g", file_pattern])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 1:  # No matches
            return f"No matches found for pattern '{pattern}'"
        if result.returncode != 0 and result.stderr:
            return f"Error: {result.stderr}"
        
        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern '{pattern}'"
        
        lines = output.split("\n")
        if len(lines) >= max_results:
            output += f"\n<results may be truncated>"
        
        return _truncate_output(output)
        
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s"
    except Exception as e:
        # Fall back to Python implementation
        return _grep(pattern, path, file_pattern, max_results, allowed_root)


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
    max_results: int = _MAX_RESULTS,
) -> str:
    """Execute a search command."""
    # Validate inputs
    if not command or not command.strip():
        return "Error: command is required"
    
    if not pattern or not pattern.strip():
        return "Error: pattern is required"
    
    # Validate path is absolute
    if not path or not path.strip():
        return "Error: path is required"
    
    if not Path(path).is_absolute():
        return f"Error: {path} is not an absolute path."
    
    # Validate max_results
    if not isinstance(max_results, int):
        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            max_results = _MAX_RESULTS
    
    if max_results <= 0 or max_results > 1000:
        max_results = _MAX_RESULTS
    
    try:
        if command == "grep":
            return _grep(pattern, path, file_pattern, max_results, _ALLOWED_ROOT)
        elif command == "find":
            return _find(pattern, path, max_results, _ALLOWED_ROOT)
        elif command == "rg":
            return _ripgrep(pattern, path, file_pattern, max_results, _ALLOWED_ROOT)
        else:
            return f"Error: unknown command '{command}'. Available commands: grep, find, rg"
    except Exception as e:
        return f"Error executing search: {e}"
