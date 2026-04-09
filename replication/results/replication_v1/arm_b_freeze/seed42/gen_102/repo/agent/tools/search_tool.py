"""
Search tool for finding files by content pattern.

Provides grep-like functionality to search within files.
"""

from __future__ import annotations

import os
import re


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "search_files",
        "description": (
            "Search for files containing a regex pattern. "
            "Returns file paths with line numbers and matching lines. "
            "Useful for finding code patterns across the codebase. "
            "Automatically skips hidden directories and common non-source folders."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search recursively",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for in file contents (case-insensitive)",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 100)",
                    "default": 100,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before/after each match (default: 0)",
                    "default": 0,
                },
            },
            "required": ["path", "pattern"],
        },
    }


def tool_function(path: str, pattern: str, file_extension: str | None = None, 
                  max_results: int = 100, context_lines: int = 0) -> str:
    """Search for files containing a pattern.

    Args:
        path: Directory path to search recursively
        pattern: Regex pattern to search for
        file_extension: Optional file extension filter (e.g., '.py')
        max_results: Maximum number of matches to return (default: 100)
        context_lines: Number of context lines to show before/after match (default: 0)

    Returns:
        String with matching files and line numbers
    """
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a valid directory"

    matches = []
    file_matches = {}  # Track matches per file for summary
    
    try:
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    for root, _, files in os.walk(path):
        # Skip hidden directories and common non-source directories
        dirs_to_skip = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
        if any(d in root for d in dirs_to_skip):
            continue
            
        for filename in files:
            if file_extension and not filename.endswith(file_extension):
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    file_match_count = 0
                    
                    for line_num, line in enumerate(lines, 1):
                        if compiled_pattern.search(line):
                            file_match_count += 1
                            
                            # Build match output with optional context
                            match_output = []
                            
                            # Add context lines before
                            if context_lines > 0:
                                start_ctx = max(0, line_num - context_lines - 1)
                                for ctx_line_num in range(start_ctx, line_num - 1):
                                    ctx_line = lines[ctx_line_num].rstrip()
                                    match_output.append(f"  {ctx_line_num + 1}: {ctx_line}")
                            
                            # Add the matching line
                            match_output.append(f"> {line_num}: {line.rstrip()}")
                            
                            # Add context lines after
                            if context_lines > 0:
                                end_ctx = min(len(lines), line_num + context_lines)
                                for ctx_line_num in range(line_num, end_ctx):
                                    ctx_line = lines[ctx_line_num].rstrip()
                                    match_output.append(f"  {ctx_line_num + 1}: {ctx_line}")
                            
                            matches.append((filepath, line_num, "\n".join(match_output)))
                            
                            if len(matches) >= max_results:
                                break
                    
                    if file_match_count > 0:
                        file_matches[filepath] = file_match_count
                        
                    if len(matches) >= max_results:
                        break
            except Exception:
                continue
        if len(matches) >= max_results:
            break

    if not matches:
        return "No matches found."

    # Build result with summary
    result_lines = [f"Found {len(matches)} match(es) in {len(file_matches)} file(s):"]
    result_lines.append("")
    
    # Add file summary
    for filepath, count in sorted(file_matches.items(), key=lambda x: -x[1])[:10]:
        result_lines.append(f"  {filepath}: {count} match(es)")
    if len(file_matches) > 10:
        result_lines.append(f"  ... and {len(file_matches) - 10} more file(s)")
    result_lines.append("")
    
    # Add detailed matches
    for filepath, line_num, match_text in matches[:50]:
        result_lines.append(f"{filepath}:{line_num}")
        result_lines.append(match_text)
        result_lines.append("")
    
    if len(matches) > 50:
        result_lines.append(f"... ({len(matches) - 50} more matches)")
    
    return "\n".join(result_lines)
