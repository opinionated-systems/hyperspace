"""
Search tool: search for patterns in files.

Provides grep-like functionality to find code patterns across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool specification for OpenAI function calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for patterns in files using grep-like functionality. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: current directory)",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., '.py', '.js')",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search is case sensitive (default: False)",
                    },
                },
                "required": ["pattern"],
            },
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_extension: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for pattern in files.

    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional extension filter (e.g., '.py')
        case_sensitive: Whether search is case sensitive

    Returns:
        String with matching lines formatted as "file:line:content"
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    flags = 0 if case_sensitive else re.IGNORECASE
    matches = []

    try:
        if os.path.isfile(path):
            # Search single file
            files_to_search = [path]
        else:
            # Search directory recursively
            files_to_search = []
            abs_path = os.path.abspath(path)
            for root, dirs, files in os.walk(abs_path):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                for filename in files:
                    if file_extension and not filename.endswith(file_extension):
                        continue
                    # Skip hidden files
                    if filename.startswith("."):
                        continue
                    files_to_search.append(os.path.join(root, filename))

        for filepath in files_to_search:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(pattern, line, flags):
                            # Truncate long lines
                            content = line.rstrip()
                            if len(content) > 200:
                                content = content[:200] + "..."
                            matches.append(f"{filepath}:{line_num}:{content}")
            except (IOError, OSError, UnicodeDecodeError):
                continue

        if not matches:
            return f"No matches found for pattern '{pattern}'"

        # Limit results to avoid overwhelming output
        if len(matches) > 100:
            return "\n".join(matches[:100]) + f"\n... ({len(matches) - 100} more matches)"

        return "\n".join(matches)

    except Exception as e:
        return f"Error during search: {str(e)}"
