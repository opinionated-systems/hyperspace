"""
Search tool for finding files and content within the codebase.

Provides grep-like functionality to search within files, find files by name,
and list directory contents.
"""

from __future__ import annotations

import os
import re
import fnmatch
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "search",
        "description": (
            "Multi-purpose search tool: (1) grep - search file contents with regex, "
            "(2) find - locate files by name pattern, (3) ls - list directory contents. "
            "Essential for exploring and understanding codebases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find", "ls"],
                    "description": "Search type: 'grep' for content, 'find' for filenames, 'ls' for directory listing",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (default: current directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "For 'grep': regex pattern to search. For 'find': filename pattern (supports * and ? wildcards). Not used for 'ls'.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js'). Only for grep and find commands.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50, max: 100)",
                    "default": 50,
                },
            },
            "required": ["command"],
        },
    }


def tool_function(
    command: str,
    path: str = ".",
    pattern: str = "",
    file_extension: str = "",
    max_results: int = 50,
) -> str:
    """Execute search command.

    Args:
        command: 'grep', 'find', or 'ls'
        path: Directory path to search
        pattern: Search pattern (regex for grep, glob for find)
        file_extension: Optional file extension filter
        max_results: Maximum results to return

    Returns:
        Search results as formatted string
    """
    max_results = min(max_results, 100)  # Cap at 100

    if command == "grep":
        return _grep_search(path, pattern, file_extension, max_results)
    elif command == "find":
        return _find_files(path, pattern, file_extension, max_results)
    elif command == "ls":
        return _list_directory(path, max_results)
    else:
        return f"Error: Unknown command '{command}'. Use 'grep', 'find', or 'ls'."


def _grep_search(directory: str, pattern: str, extension: str, max_results: int) -> str:
    """Search file contents with regex pattern."""
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found"
    if not pattern:
        return "Error: Pattern required for grep search"

    try:
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    matches = []
    files_searched = 0

    for root, dirs, files in os.walk(directory):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]

        for filename in files:
            if filename.startswith("."):
                continue
            if extension and not filename.endswith(extension):
                continue

            filepath = os.path.join(root, filename)
            files_searched += 1

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    lines = content.splitlines()
                    for line_num, line in enumerate(lines, 1):
                        if compiled_pattern.search(line):
                            matches.append(f"{filepath}:{line_num}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                break
                    if len(matches) >= max_results:
                        break
            except (IOError, OSError, UnicodeDecodeError):
                continue

        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches for pattern '{pattern}' in '{directory}' (searched {files_searched} files)"

    header = f"Found {len(matches)} matches in {files_searched} files searched:\n"
    return header + "\n".join(matches)


def _find_files(directory: str, pattern: str, extension: str, max_results: int) -> str:
    """Find files by name pattern (supports wildcards)."""
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found"

    matches = []
    files_checked = 0

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]

        for filename in files:
            if filename.startswith("."):
                continue
            if extension and not filename.endswith(extension):
                continue

            files_checked += 1

            # Pattern matching: exact, substring, or glob
            if pattern:
                if "*" in pattern or "?" in pattern:
                    if not fnmatch.fnmatch(filename.lower(), pattern.lower()):
                        continue
                elif pattern.lower() not in filename.lower():
                    continue

            matches.append(os.path.join(root, filename))
            if len(matches) >= max_results:
                break

        if len(matches) >= max_results:
            break

    if not matches:
        msg = f"No files found in '{directory}'"
        if pattern:
            msg += f" matching '{pattern}'"
        if extension:
            msg += f" with extension '{extension}'"
        return msg + f" (checked {files_checked} files)"

    header = f"Found {len(matches)} files:\n"
    return header + "\n".join(matches)


def _list_directory(directory: str, max_results: int) -> str:
    """List directory contents with file sizes and types."""
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found"

    try:
        entries = os.listdir(directory)
    except PermissionError:
        return f"Error: Permission denied for '{directory}'"

    # Separate and sort
    dirs = []
    files = []

    for entry in sorted(entries, key=str.lower):
        if entry.startswith(".") or entry in ("__pycache__", "node_modules"):
            continue

        full_path = os.path.join(directory, entry)
        try:
            if os.path.isdir(full_path):
                dirs.append(entry)
            else:
                size = os.path.getsize(full_path)
                size_str = _format_size(size)
                files.append((entry, size_str))
        except (OSError, IOError):
            continue

    # Build output
    lines = [f"Contents of '{directory}':"]
    lines.append("-" * 50)

    # Directories first
    for d in dirs[:max_results]:
        lines.append(f"📁 {d}/")

    # Then files
    remaining = max_results - len(dirs)
    for f, size in files[:remaining]:
        lines.append(f"📄 {f} ({size})")

    total = len(dirs) + len(files)
    shown = min(len(dirs), max_results) + min(len(files), remaining)
    if shown < total:
        lines.append(f"\n... and {total - shown} more entries")

    lines.append(f"\nTotal: {len(dirs)} directories, {len(files)} files")
    return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
