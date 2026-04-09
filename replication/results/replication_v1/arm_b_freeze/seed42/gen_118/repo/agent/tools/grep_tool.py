"""
Grep tool: search for patterns within file contents.

Provides grep-like functionality to search for text patterns in files,
with support for regex, case sensitivity options, and file filtering.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "grep",
        "description": (
            "Search for patterns within file contents. "
            "Supports regex patterns, case sensitivity options, and file filtering. "
            "Returns matching lines with file paths and line numbers."
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
                    "description": "Absolute path to directory or file to search in.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: False).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Search for pattern in files."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        results = []
        result_count = 0

        # Determine files to search
        if p.is_file():
            files = [p]
        elif p.is_dir():
            if file_pattern:
                files = list(p.rglob(file_pattern))
            else:
                # Default: search common code files
                files = []
                for ext in [".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".md", ".txt", ".json", ".yaml", ".yml"]:
                    files.extend(p.rglob(f"*{ext}"))
                # Also include files without extension that might be scripts
                for f in p.rglob("*"):
                    if f.is_file() and "." not in f.name:
                        files.append(f)
        else:
            return f"Error: {path} does not exist."

        # Search in files
        for file_path in files:
            if not file_path.is_file():
                continue
                
            # Skip binary files and hidden files
            if file_path.name.startswith("."):
                continue
            if ".pyc" in file_path.suffix or file_path.suffix in [".exe", ".dll", ".so", ".dylib"]:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Format: file_path:line_num: content
                            results.append(f"{file_path}:{line_num}: {line.rstrip()}")
                            result_count += 1
                            if result_count >= max_results:
                                break
                        if result_count >= max_results:
                            break
            except (IOError, OSError, PermissionError):
                continue

        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"

        header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
        return header + "\n".join(results[:max_results])

    except Exception as e:
        return f"Error: {e}"
