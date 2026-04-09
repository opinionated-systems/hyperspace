"""
File search tool: search for files by name or content pattern.

Provides grep-like functionality to find files matching patterns.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_search",
        "description": (
            "Search for files by name pattern or content pattern. "
            "Uses find and grep for efficient searching. "
            "Results are truncated to avoid overwhelming output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename glob or regex for content search).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["filename", "content"],
                    "description": "Type of search: 'filename' for file names, 'content' for file contents.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path). Defaults to allowed root.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default 50.",
                },
            },
            "required": ["pattern", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a file search."""
    try:
        # Determine search root
        if path is not None:
            search_path = Path(path)
            if not search_path.is_absolute():
                return f"Error: {path} is not an absolute path."
        elif _ALLOWED_ROOT is not None:
            search_path = Path(_ALLOWED_ROOT)
        else:
            search_path = Path(os.getcwd())

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(search_path))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if not search_path.exists():
            return f"Error: {search_path} does not exist."

        if not search_path.is_dir():
            return f"Error: {search_path} is not a directory."

        results = []

        if search_type == "filename":
            # Use find to search for files by name pattern
            cmd = [
                "find", str(search_path),
                "-type", "f",
                "-name", pattern,
                "-not", "-path", "*/\.*",  # Exclude hidden files
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    files = [f for f in result.stdout.strip().split("\n") if f]
                    results = files[:max_results]
                else:
                    return f"Error running find: {result.stderr}"
            except subprocess.TimeoutExpired:
                return "Error: search timed out after 30 seconds"
            except Exception as e:
                return f"Error: {e}"

        elif search_type == "content":
            # Use grep to search for content pattern
            cmd = [
                "grep", "-r", "-l",
                "--include=*.py",
                "--include=*.txt",
                "--include=*.md",
                "--include=*.json",
                "--include=*.yaml",
                "--include=*.yml",
                pattern, str(search_path),
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                # grep returns 0 if matches found, 1 if no matches, >1 for errors
                if result.returncode in (0, 1):
                    files = [f for f in result.stdout.strip().split("\n") if f]
                    results = files[:max_results]
                else:
                    return f"Error running grep: {result.stderr}"
            except subprocess.TimeoutExpired:
                return "Error: search timed out after 30 seconds"
            except Exception as e:
                return f"Error: {e}"
        else:
            return f"Error: unknown search_type '{search_type}'. Use 'filename' or 'content'."

        # Format results
        if not results:
            return f"No files found matching '{pattern}' (search_type={search_type})"

        output = f"Found {len(results)} file(s) matching '{pattern}':\n"
        if len(results) == max_results:
            output = f"Found {len(results)}+ file(s) matching '{pattern}' (showing first {max_results}):\n"

        for i, f in enumerate(results, 1):
            # Show relative path if under allowed root
            if _ALLOWED_ROOT and f.startswith(_ALLOWED_ROOT):
                rel_path = f[len(_ALLOWED_ROOT):].lstrip("/")
                output += f"  {i}. {rel_path}\n"
            else:
                output += f"  {i}. {f}\n"

        return output.strip()

    except Exception as e:
        return f"Error: {e}"
