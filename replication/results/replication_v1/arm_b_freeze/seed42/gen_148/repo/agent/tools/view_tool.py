"""
View tool: view file contents with line range support.

Complements the editor tool by allowing efficient viewing of large files
without loading the entire content. Supports viewing specific line ranges.
"""

from __future__ import annotations


def tool_info() -> dict:
    return {
        "name": "view",
        "description": "View the contents of a file. Supports viewing specific line ranges for large files. If the file is over 200 lines, use view_range to view specific sections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to view",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional [start_line, end_line] to view a specific range (1-indexed, inclusive). If omitted, shows the whole file or first 100 lines for large files.",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str, view_range: list[int] | None = None) -> str:
    """View file contents, optionally with line range."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading file: {type(e).__name__}: {e}"

    total_lines = len(lines)
    
    if total_lines == 0:
        return f"File is empty: {path}"
    
    # Determine range to show
    if view_range is not None:
        start, end = view_range
        # Convert to 0-indexed
        start_idx = max(0, start - 1)
        end_idx = min(total_lines, end)
        if start_idx >= end_idx:
            return f"Error: Invalid range [{start}, {end}] for file with {total_lines} lines"
        selected_lines = lines[start_idx:end_idx]
        header = f"Viewing lines {start}-{min(end, total_lines)} of {total_lines} in {path}\n"
    else:
        # No range specified - show all or first 100 lines
        if total_lines <= 100:
            selected_lines = lines
            header = f"Viewing all {total_lines} lines in {path}\n"
        else:
            selected_lines = lines[:100]
            header = f"Viewing first 100 lines of {total_lines} in {path} (use view_range to see more)\n"
    
    # Format with line numbers
    start_line_num = view_range[0] if view_range else 1
    numbered_lines = []
    for i, line in enumerate(selected_lines):
        line_num = start_line_num + i
        # Remove trailing newline for formatting
        content = line.rstrip('\n\r')
        numbered_lines.append(f"{line_num:4d} | {content}")
    
    return header + "\n".join(numbered_lines)
