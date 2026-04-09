"""
Diff tool: compare files and show differences.

Useful for understanding what changes have been made between
file versions or different files.
"""

from __future__ import annotations

import difflib
from pathlib import Path


def _read_file(path: str) -> str:
    """Read file contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"


def diff_files(file1: str, file2: str, context_lines: int = 3) -> str:
    """Compare two files and show differences.
    
    Args:
        file1: Path to first file
        file2: Path to second file
        context_lines: Number of context lines in diff (default: 3)
    
    Returns:
        Unified diff showing differences between files
    """
    content1 = _read_file(file1)
    content2 = _read_file(file2)
    
    if content1.startswith("Error"):
        return content1
    if content2.startswith("Error"):
        return content2
    
    lines1 = content1.splitlines(keepends=True)
    lines2 = content2.splitlines(keepends=True)
    
    # Ensure lines end with newline for proper diff
    if lines1 and not lines1[-1].endswith("\n"):
        lines1[-1] += "\n"
    if lines2 and not lines2[-1].endswith("\n"):
        lines2[-1] += "\n"
    
    diff = difflib.unified_diff(
        lines1,
        lines2,
        fromfile=file1,
        tofile=file2,
        n=context_lines,
    )
    
    result = "".join(diff)
    if not result:
        return f"Files are identical: {file1} and {file2}"
    
    return result


def diff_strings(str1: str, str2: str, label1: str = "original", label2: str = "modified") -> str:
    """Compare two strings and show differences.
    
    Args:
        str1: First string
        str2: Second string
        label1: Label for first string
        label2: Label for second string
    
    Returns:
        Unified diff showing differences
    """
    lines1 = str1.splitlines(keepends=True)
    lines2 = str2.splitlines(keepends=True)
    
    # Ensure lines end with newline
    if lines1 and not lines1[-1].endswith("\n"):
        lines1[-1] += "\n"
    if lines2 and not lines2[-1].endswith("\n"):
        lines2[-1] += "\n"
    
    diff = difflib.unified_diff(
        lines1,
        lines2,
        fromfile=label1,
        tofile=label2,
        n=3,
    )
    
    result = "".join(diff)
    if not result:
        return "Strings are identical"
    
    return result


def tool_info() -> dict:
    return {
        "name": "diff",
        "description": "Compare files or strings and show differences. Useful for understanding what changes have been made.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["files", "strings"],
                    "description": "Comparison mode: 'files' to compare two files, 'strings' to compare two strings",
                },
                "file1": {
                    "type": "string",
                    "description": "Path to first file (required for mode='files')",
                },
                "file2": {
                    "type": "string",
                    "description": "Path to second file (required for mode='files')",
                },
                "str1": {
                    "type": "string",
                    "description": "First string (required for mode='strings')",
                },
                "str2": {
                    "type": "string",
                    "description": "Second string (required for mode='strings')",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines in diff output (default: 3)",
                    "default": 3,
                },
            },
            "required": ["mode"],
        },
    }


def tool_function(mode: str, file1: str = "", file2: str = "", str1: str = "", str2: str = "", context_lines: int = 3) -> str:
    """Execute diff tool."""
    if mode == "files":
        if not file1 or not file2:
            return "Error: file1 and file2 are required for mode='files'"
        return diff_files(file1, file2, context_lines)
    elif mode == "strings":
        return diff_strings(str1, str2)
    else:
        return f"Error: Unknown mode '{mode}'. Use 'files' or 'strings'."
