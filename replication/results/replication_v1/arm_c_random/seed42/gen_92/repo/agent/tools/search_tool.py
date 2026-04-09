"""
Search tool: search for files and content within the repository.

Provides file and content search capabilities to help navigate codebases.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata matching the expected schema."""
    return {
        "name": "search",
        "description": "Search for files or content within a directory. Can search by filename pattern or content pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search in",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to match filenames (e.g., '*.py', '*.txt')",
                },
                "content_pattern": {
                    "type": "string",
                    "description": "Optional regex pattern to search within file contents",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20)",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(
    path: str,
    file_pattern: str | None = None,
    content_pattern: str | None = None,
    max_results: int = 20,
) -> str:
    """Search for files or content within a directory.
    
    Args:
        path: Directory path to search in
        file_pattern: Optional glob pattern to match filenames
        content_pattern: Optional regex pattern to search within file contents
        max_results: Maximum number of results to return
        
    Returns:
        Search results as a formatted string
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a directory"
    
    results = []
    search_path = Path(path)
    
    try:
        # Collect files to search
        if file_pattern:
            files = list(search_path.rglob(file_pattern))
        else:
            files = [f for f in search_path.rglob("*") if f.is_file()]
        
        # Filter by content pattern if provided
        if content_pattern:
            regex = re.compile(content_pattern, re.IGNORECASE)
            matching_files = []
            
            for file_path in files:
                if len(matching_files) >= max_results:
                    break
                    
                try:
                    # Skip binary files and very large files
                    if file_path.stat().st_size > 1_000_000:  # 1MB limit
                        continue
                        
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        
                    if regex.search(content):
                        # Find line numbers for matches
                        lines = content.split("\n")
                        match_lines = []
                        for i, line in enumerate(lines, 1):
                            if regex.search(line):
                                match_lines.append(i)
                                if len(match_lines) >= 3:  # Limit line matches per file
                                    break
                        
                        matching_files.append({
                            "path": str(file_path.relative_to(search_path)),
                            "lines": match_lines,
                        })
                except (IOError, OSError):
                    continue
            
            results = matching_files
        else:
            # Just return file paths
            results = [{"path": str(f.relative_to(search_path))} for f in files[:max_results]]
        
        # Format results
        if not results:
            return "No matches found."
        
        output_lines = [f"Found {len(results)} result(s):"]
        for r in results:
            line_info = ""
            if "lines" in r:
                line_info = f" (lines: {', '.join(map(str, r['lines']))})"
            output_lines.append(f"  - {r['path']}{line_info}")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"Error during search: {e}"
