"""
Search tool: search for text patterns in files.

Provides grep-like functionality to find text patterns across files.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files. "
            "Supports regex patterns and can search within specific directories. "
            "Returns matching file paths with line numbers and context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default 50.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files."""
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: no path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)

        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        # Build find command
        if file_extension:
            find_cmd = ["find", search_path, "-type", "f", "-name", f"*{file_extension}"]
        else:
            find_cmd = ["find", search_path, "-type", "f"]

        # Exclude hidden directories and __pycache__
        find_cmd.extend(["-not", "-path", "*/\.*", "-not", "-path", "*/__pycache__/*"])

        # Run find to get files
        find_result = subprocess.run(
            find_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if find_result.returncode != 0:
            return f"Error running find: {find_result.stderr}"

        files = find_result.stdout.strip().split("\n") if find_result.stdout.strip() else []

        if not files or files == [""]:
            return f"No files found in {search_path}"

        # Search in files with grep
        results = []
        count = 0

        for file_path in files:
            if not file_path:
                continue
            if count >= max_results:
                results.append(f"... (truncated at {max_results} results)")
                break

            try:
                # Use grep with line numbers
                grep_result = subprocess.run(
                    ["grep", "-n", "-H", "-E", pattern, file_path],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if grep_result.returncode == 0 and grep_result.stdout:
                    lines = grep_result.stdout.strip().split("\n")
                    for line in lines:
                        if count >= max_results:
                            break
                        results.append(line)
                        count += 1
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue

        if not results:
            return f"No matches found for pattern '{pattern}' in {search_path}"

        return f"Found {count} match(es) for '{pattern}':\n" + "\n".join(results)

    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except Exception as e:
        return f"Error: {e}"
