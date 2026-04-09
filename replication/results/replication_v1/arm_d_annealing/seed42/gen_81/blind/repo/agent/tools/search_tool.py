"""
Search tool: search for files and content within the repository.

Provides grep-like functionality to find files by name or content.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files by name pattern or search file contents for text patterns. Uses find and grep commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["find", "grep"],
                        "description": "Type of search: 'find' for file names, 'grep' for content within files",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (relative or absolute)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern - filename pattern for 'find', regex pattern for 'grep'",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., '.py', '.txt'). Only used with grep command.",
                    },
                },
                "required": ["command", "path", "pattern"],
            },
        },
    }


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_extension: str | None = None,
) -> dict[str, Any]:
    """Execute search command.

    Args:
        command: 'find' for file names, 'grep' for content
        path: Directory to search in
        pattern: Search pattern
        file_extension: Optional extension filter for grep

    Returns:
        Dict with output, error, and return_code
    """
    # Validate and sanitize path
    if not os.path.exists(path):
        return {
            "output": "",
            "error": f"Path does not exist: {path}",
            "return_code": 1,
        }

    if command == "find":
        # Find files by name pattern
        try:
            result = subprocess.run(
                ["find", path, "-type", "f", "-name", pattern],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip()
            files = output.split("\n") if output else []
            return {
                "output": files,
                "error": result.stderr if result.returncode != 0 else "",
                "return_code": result.returncode,
                "count": len(files),
            }
        except subprocess.TimeoutExpired:
            return {
                "output": [],
                "error": "Search timed out after 30 seconds",
                "return_code": 1,
                "count": 0,
            }
        except Exception as e:
            return {
                "output": [],
                "error": str(e),
                "return_code": 1,
                "count": 0,
            }

    elif command == "grep":
        # Search file contents
        try:
            # Build grep command with optional file extension filter
            if file_extension:
                include_pattern = f"*{file_extension}"
                cmd = [
                    "grep", "-r", "-l", "-i",  # recursive, list filenames only, case-insensitive
                    "--include", include_pattern,
                    pattern, path
                ]
            else:
                cmd = [
                    "grep", "-r", "-l", "-i",  # recursive, list filenames only, case-insensitive
                    pattern, path
                ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip()
            files = output.split("\n") if output else []
            return {
                "output": files,
                "error": result.stderr if result.returncode != 0 else "",
                "return_code": result.returncode,
                "count": len(files),
            }
        except subprocess.TimeoutExpired:
            return {
                "output": [],
                "error": "Search timed out after 30 seconds",
                "return_code": 1,
                "count": 0,
            }
        except Exception as e:
            return {
                "output": [],
                "error": str(e),
                "return_code": 1,
                "count": 0,
            }

    else:
        return {
            "output": [],
            "error": f"Unknown command: {command}. Use 'find' or 'grep'.",
            "return_code": 1,
            "count": 0,
        }
