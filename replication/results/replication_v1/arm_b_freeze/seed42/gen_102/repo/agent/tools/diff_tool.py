"""
Diff tool: compare files or text to show differences.

Provides unified diff output for comparing file versions or
text content, useful for reviewing changes before applying them.
"""

from __future__ import annotations

import difflib
import os


def tool_info() -> dict:
    return {
        "name": "diff",
        "description": "Compare two files or text strings and show differences in unified diff format. Useful for reviewing changes before applying them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "original": {
                    "type": "string",
                    "description": "Original text content or file path",
                },
                "modified": {
                    "type": "string",
                    "description": "Modified text content or file path",
                },
                "original_is_path": {
                    "type": "boolean",
                    "description": "Whether 'original' is a file path (default: False)",
                    "default": False,
                },
                "modified_is_path": {
                    "type": "boolean",
                    "description": "Whether 'modified' is a file path (default: False)",
                    "default": False,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show (default: 3)",
                    "default": 3,
                },
            },
            "required": ["original", "modified"],
        },
    }


def _read_file_if_path(content: str, is_path: bool) -> tuple[str, str]:
    """Read file if content is a path, otherwise return content as-is."""
    if is_path:
        if not os.path.isfile(content):
            return "", f"Error: File not found: {content}"
        try:
            with open(content, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(), content
        except Exception as e:
            return "", f"Error reading file: {e}"
    return content, "<string>"


def tool_function(
    original: str,
    modified: str,
    original_is_path: bool = False,
    modified_is_path: bool = False,
    context_lines: int = 3,
) -> str:
    """Compare two texts or files and return unified diff."""
    
    # Read content
    original_content, original_name = _read_file_if_path(original, original_is_path)
    if original_content.startswith("Error:"):
        return original_content
        
    modified_content, modified_name = _read_file_if_path(modified, modified_is_path)
    if modified_content.startswith("Error:"):
        return modified_content
    
    # Split into lines
    original_lines = original_content.splitlines(keepends=True)
    modified_lines = modified_content.splitlines(keepends=True)
    
    # Ensure lines end with newline for proper diff
    if original_lines and not original_lines[-1].endswith('\n'):
        original_lines[-1] += '\n'
    if modified_lines and not modified_lines[-1].endswith('\n'):
        modified_lines[-1] += '\n'
    
    # Generate unified diff
    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=original_name,
        tofile=modified_name,
        n=context_lines,
    )
    
    diff_text = "".join(diff)
    
    if not diff_text:
        return "No differences found - files are identical."
    
    # Count changes
    added = sum(1 for line in diff_text.split('\n') if line.startswith('+') and not line.startswith('+++'))
    removed = sum(1 for line in diff_text.split('\n') if line.startswith('-') and not line.startswith('---'))
    
    summary = f"Diff summary: {added} line(s) added, {removed} line(s) removed\n"
    
    return summary + "\n" + diff_text
