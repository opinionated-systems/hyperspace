"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns
within the allowed root directory.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files. "
            "Supports text search, regex patterns, and file filtering. "
            "Results include file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (text or regex).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py'). Optional.",
                },
                "use_regex": {
                    "type": "boolean",
                    "description": "Whether to treat pattern as regex. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
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


def _truncate_results(results: list[str], max_results: int) -> str:
    """Truncate results if too many."""
    if len(results) > max_results:
        shown = results[:max_results]
        hidden_count = len(results) - max_results
        return "\n".join(shown) + f"\n... ({hidden_count} more results truncated)"
    return "\n".join(results)


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    use_regex: bool = False,
    max_results: int = 50,
) -> str:
    """Search for pattern in files under the given path."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        results: list[str] = []

        if p.is_file():
            # Search single file
            files_to_search = [p]
        else:
            # Find files to search
            if file_pattern:
                files_to_search = list(p.rglob(file_pattern))
            else:
                # Default: search common text file types, skip hidden dirs
                files_to_search = [
                    f for f in p.rglob("*")
                    if f.is_file()
                    and not any(part.startswith(".") for part in f.parts)
                    and f.suffix in (".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".js", ".ts", ".html", ".css", ".rs", ".go", ".java", ".c", ".cpp", ".h")
                ]

        # Compile regex if needed
        if use_regex:
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return f"Error: invalid regex pattern: {e}"
        else:
            regex = None

        # Search files
        for file_path in files_to_search:
            try:
                content = file_path.read_text(errors="ignore")
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    if use_regex:
                        if regex.search(line):
                            results.append(f"{file_path}:{i}:{line[:100]}")
                    else:
                        if pattern in line:
                            results.append(f"{file_path}:{i}:{line[:100]}")

                    if len(results) > max_results * 2:  # Early cutoff
                        break

            except Exception:
                continue

        if not results:
            return f"No matches found for '{pattern}' in {path}"

        return f"Found {len(results)} matches:\n" + _truncate_results(results, max_results)

    except Exception as e:
        return f"Error: {e}"
