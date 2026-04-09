"""
File statistics tool: Get detailed statistics about files.

Provides line count, word count, character count, and file size information.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_stats",
            "description": "Get detailed statistics about a file including line count, word count, character count, and file size. Works with text files, code files, and any readable file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to analyze",
                    },
                },
                "required": ["path"],
            },
        },
    }


def tool_function(path: str) -> str:
    """Get statistics about a file.

    Args:
        path: Absolute path to the file

    Returns:
        Formatted string with file statistics
    """
    try:
        file_path = Path(path)
        
        if not file_path.exists():
            return f"Error: File not found: {path}"
        
        if not file_path.is_file():
            return f"Error: Path is not a file: {path}"
        
        # Get file size
        size_bytes = file_path.stat().st_size
        size_kb = size_bytes / 1024
        
        # Read file content for text statistics
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            line_count = len(lines)
            non_empty_lines = len([l for l in lines if l.strip()])
            word_count = len(content.split())
            char_count = len(content)
            char_count_no_spaces = len(content.replace(' ', '').replace('\n', '').replace('\t', ''))
            
            stats = f"""File Statistics for: {path}
================================
Size: {size_bytes:,} bytes ({size_kb:.2f} KB)
Lines: {line_count:,} (non-empty: {non_empty_lines:,})
Words: {word_count:,}
Characters: {char_count:,} (without whitespace: {char_count_no_spaces:,})
Average words per line: {word_count / line_count:.1f}"""
            
            return stats
            
        except Exception as e:
            # Binary or unreadable file - return basic stats only
            return f"""File Statistics for: {path}
================================
Size: {size_bytes:,} bytes ({size_kb:.2f} KB)
Note: File appears to be binary or non-text format.
Error reading content: {str(e)}"""
            
    except Exception as e:
        return f"Error analyzing file: {str(e)}"
