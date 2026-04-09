"""
Search tool: search for files and content within the repository.

Provides capabilities to:
1. Search for files by name pattern
2. Search for content within files
3. Find files containing specific text
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata for search operations."""
    return {
        "name": "search",
        "description": "Search for files and content within the repository. Supports searching by filename pattern (e.g., '*.py') or content within files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search in (relative or absolute)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Filename pattern to match (e.g., '*.py', 'test_*.py'). Uses shell-style wildcards.",
                },
                "content": {
                    "type": "string",
                    "description": "Optional text content to search for within files. If provided, only files containing this text will be returned.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories. Default: true",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(
    path: str,
    pattern: str | None = None,
    content: str | None = None,
    recursive: bool = True,
) -> str:
    """Search for files matching criteria.

    Args:
        path: Directory path to search in
        pattern: Optional filename pattern (e.g., '*.py')
        content: Optional text to search for within files
        recursive: Whether to search subdirectories

    Returns:
        String with search results
    """
    try:
        search_path = Path(path).expanduser().resolve()
        if not search_path.exists():
            return f"Error: Path '{path}' does not exist"
        if not search_path.is_dir():
            return f"Error: Path '{path}' is not a directory"

        results = []
        files_searched = 0
        matches_found = 0

        # Determine walk strategy
        if recursive:
            walk_iter = os.walk(search_path)
        else:
            # Only top-level
            try:
                entries = list(os.scandir(search_path))
                files = [e for e in entries if e.is_file()]
                dirs = []
                walk_iter = [(str(search_path), dirs, [f.name for f in files])]
            except OSError as e:
                return f"Error reading directory: {e}"

        for root, _dirs, files in walk_iter:
            for filename in files:
                # Check pattern match
                if pattern and not fnmatch.fnmatch(filename, pattern):
                    continue

                filepath = Path(root) / filename
                files_searched += 1

                # Check content match if requested
                if content:
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            file_content = f.read()
                            if content not in file_content:
                                continue
                            # Find line numbers
                            lines = file_content.split("\n")
                            matching_lines = []
                            for i, line in enumerate(lines, 1):
                                if content in line:
                                    matching_lines.append(f"  Line {i}: {line.strip()[:100]}")
                            results.append(f"{filepath}\n" + "\n".join(matching_lines[:5]))
                            matches_found += 1
                    except (OSError, UnicodeDecodeError):
                        continue
                else:
                    results.append(str(filepath))
                    matches_found += 1

                # Limit results to avoid overwhelming output
                if matches_found >= 50:
                    results.append(f"... (truncated, showing first 50 matches)")
                    break

            if matches_found >= 50:
                break

        if not results:
            if content:
                return f"No files found containing '{content}' in '{path}'"
            elif pattern:
                return f"No files matching pattern '{pattern}' found in '{path}'"
            else:
                return f"No files found in '{path}'"

        summary = f"Found {matches_found} match(es) in {files_searched} file(s) searched:\n\n"
        return summary + "\n\n".join(results)

    except Exception as e:
        return f"Error during search: {e}"
