"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns
within files in the allowed root directory.
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
            "Supports regex patterns and file filtering. "
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
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default True).",
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
    file_extension: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = True,
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
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        results = []
        count = 0

        # Compile regex with appropriate flags
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error:
            # Invalid regex, treat as literal string
            compiled_pattern = None

        if p.is_file():
            # Search in single file
            files_to_search = [p]
        else:
            # Search in directory
            if file_extension:
                files_to_search = list(p.rglob(f"*{file_extension}"))
            else:
                files_to_search = list(p.rglob("*"))
            # Filter to files only
            files_to_search = [f for f in files_to_search if f.is_file()]

        for file_path in files_to_search:
            if count >= max_results:
                break

            # Skip binary files and hidden files
            if file_path.name.startswith("."):
                continue
            if ".pyc" in file_path.suffix or file_path.suffix == ".pyc":
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    if count >= max_results:
                        break

                    try:
                        if compiled_pattern is not None:
                            if compiled_pattern.search(line):
                                # Format: file_path:line_num: line_content
                                preview = line.strip()[:100]
                                results.append(f"{file_path}:{line_num}: {preview}")
                                count += 1
                        else:
                            # Literal string search with case sensitivity
                            if case_sensitive:
                                match = pattern in line
                            else:
                                match = pattern.lower() in line.lower()
                            if match:
                                preview = line.strip()[:100]
                                results.append(f"{file_path}:{line_num}: {preview}")
                                count += 1
                    except re.error:
                        # Invalid regex, treat as literal string
                        if case_sensitive:
                            match = pattern in line
                        else:
                            match = pattern.lower() in line.lower()
                        if match:
                            preview = line.strip()[:100]
                            results.append(f"{file_path}:{line_num}: {preview}")
                            count += 1

            except (IOError, OSError, UnicodeDecodeError):
                # Skip files we can't read
                continue

        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"

        header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
        return header + "\n".join(results)

    except Exception as e:
        return f"Error: {e}"
