"""
File tool: search and explore files in the filesystem.

Provides grep-like search and file listing capabilities.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool specification for file operations."""
    return {
        "name": "file_search",
        "description": "Search for files and content within a directory. Provides grep-like search and file listing capabilities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "list"],
                    "description": "The file operation to perform: 'grep' to search file contents, 'list' to list files in a directory",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (for list) or file/directory path to search within (for grep)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern for grep command (required when command='grep')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively (for grep and list commands)",
                    "default": True,
                },
            },
            "required": ["command", "path"],
        },
    }


def _validate_path(path: str) -> str | None:
    """Validate and normalize path, returning error message if invalid."""
    if not path:
        return "Error: path cannot be empty"
    # Expand home directory
    expanded = os.path.expanduser(path)
    # Normalize path
    normalized = os.path.normpath(expanded)
    return None


def _do_grep(path: str, pattern: str, recursive: bool) -> str:
    """Execute grep search."""
    error = _validate_path(path)
    if error:
        return error

    if not pattern:
        return "Error: pattern is required for grep command"

    expanded = os.path.expanduser(path)

    # Check if path exists
    if not os.path.exists(expanded):
        return f"Error: path '{path}' does not exist"

    try:
        # Build grep command
        cmd = ["grep", "-n", "-I"]  # -n for line numbers, -I to skip binary files
        if recursive:
            cmd.append("-r")
        cmd.append(pattern)
        cmd.append(expanded)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Found matches
            lines = result.stdout.strip().split("\n")
            # Limit output to avoid overwhelming responses
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return result.stdout.strip()
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for '{pattern}' in '{path}'"
        else:
            # Error occurred
            return f"Error: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "Error: grep command timed out (30s limit)"
    except Exception as e:
        return f"Error executing grep: {e}"


def _do_list(path: str, recursive: bool) -> str:
    """List files in directory."""
    error = _validate_path(path)
    if error:
        return error

    expanded = os.path.expanduser(path)

    # Check if path exists and is a directory
    if not os.path.exists(expanded):
        return f"Error: path '{path}' does not exist"
    if not os.path.isdir(expanded):
        return f"Error: path '{path}' is not a directory"

    try:
        if recursive:
            # Walk directory tree
            lines = []
            for root, dirs, files in os.walk(expanded):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                level = root.replace(expanded, "").count(os.sep)
                indent = "  " * level
                rel_path = os.path.relpath(root, expanded)
                if rel_path == ".":
                    lines.append(f"{os.path.basename(expanded)}/")
                else:
                    lines.append(f"{indent}{os.path.basename(root)}/")
                sub_indent = "  " * (level + 1)
                for file in sorted(files):
                    if not file.startswith("."):
                        lines.append(f"{sub_indent}{file}")
            return "\n".join(lines) if lines else f"Directory '{path}' is empty"
        else:
            # List only immediate contents
            entries = os.listdir(expanded)
            dirs = sorted([e for e in entries if os.path.isdir(os.path.join(expanded, e)) and not e.startswith(".")])
            files = sorted([e for e in entries if os.path.isfile(os.path.join(expanded, e)) and not e.startswith(".")])

            lines = []
            for d in dirs:
                lines.append(f"{d}/")
            for f in files:
                lines.append(f)

            return "\n".join(lines) if lines else f"Directory '{path}' is empty"

    except Exception as e:
        return f"Error listing directory: {e}"


def tool_function(command: str, path: str, pattern: str = "", recursive: bool = True) -> str:
    """Execute file search operation.

    Args:
        command: The operation to perform ('grep' or 'list')
        path: Path to operate on
        pattern: Search pattern (for grep command)
        recursive: Whether to operate recursively

    Returns:
        Result of the file operation
    """
    if command == "grep":
        return _do_grep(path, pattern, recursive)
    elif command == "list":
        return _do_list(path, recursive)
    else:
        return f"Error: unknown command '{command}'"
