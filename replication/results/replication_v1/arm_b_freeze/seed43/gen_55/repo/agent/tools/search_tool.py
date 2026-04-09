"""
Search tool: search for patterns in files.

Provides grep-like functionality for finding code patterns,
useful for navigating larger codebases.
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
            "Search for patterns in files using grep or regex. "
            "Supports searching file contents and filenames. "
            "Results are truncated to prevent context overflow."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex or literal string).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Default: allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to search (e.g., '*.py'). Default: all files.",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Whether pattern is a regex. Default: false (literal search).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
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
    file_pattern: str | None = None,
    is_regex: bool = False,
    max_results: int = 50,
) -> str:
    """Search for pattern in files."""
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No search path provided and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if not os.path.exists(search_path):
            return f"Error: path does not exist: {search_path}"

        results = []
        count = 0

        # Build file list
        if os.path.isfile(search_path):
            files = [Path(search_path)]
        else:
            p = Path(search_path)
            if file_pattern:
                files = list(p.rglob(file_pattern))
            else:
                files = [f for f in p.rglob("*") if f.is_file()]

        # Compile regex if needed
        if is_regex:
            try:
                compiled = re.compile(pattern)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"
        else:
            compiled = None

        # Search files
        for f in files:
            if count >= max_results:
                break

            # Skip binary files and hidden files
            if f.name.startswith(".") or "/." in str(f):
                continue

            try:
                content = f.read_text(errors="ignore")
            except (IOError, OSError):
                continue

            # Search for pattern
            matches = []
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                if is_regex:
                    if compiled.search(line):
                        matches.append((i, line.strip()))
                else:
                    if pattern in line:
                        matches.append((i, line.strip()))

            if matches:
                count += 1
                rel_path = f.relative_to(search_path) if f.is_relative_to(search_path) else f
                result_lines = [f"  {ln}: {ln_text[:100]}" for ln, ln_text in matches[:5]]
                if len(matches) > 5:
                    result_lines.append(f"  ... and {len(matches) - 5} more matches")
                results.append(f"{rel_path}:\n" + "\n".join(result_lines))

        if not results:
            return f"No matches found for '{pattern}'"

        output = f"Found {count} file(s) with matches:\n\n" + "\n\n".join(results)
        if len(output) > 8000:
            output = output[:4000] + "\n... [output truncated] ...\n" + output[-4000:]

        return output

    except Exception as e:
        return f"Error: {e}"
