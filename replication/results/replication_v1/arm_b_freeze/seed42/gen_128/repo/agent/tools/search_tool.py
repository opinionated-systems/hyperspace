"""
Search tool: search for patterns in files using grep-like functionality.

Provides capabilities to:
- Search for text patterns in files
- Search within specific file extensions
- Search recursively in directories
- Get context lines around matches
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for patterns in files. Supports regex and glob patterns for file filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The search pattern (regex supported)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Default is current directory.",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.py', '*.js'). Default searches all files.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively in subdirectories. Default is True.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search is case sensitive. Default is False.",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines to show before and after each match. Default is 2.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default is 50.",
                    },
                },
                "required": ["pattern"],
            },
        },
    }


def _read_file_with_context(
    file_path: Path, 
    pattern: str, 
    case_sensitive: bool,
    context_lines: int,
    max_results: int,
) -> list[dict]:
    """Read a file and find matches with context."""
    results = []
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled_pattern = re.compile(pattern, flags)
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return results
    
    for i, line in enumerate(lines):
        if compiled_pattern.search(line):
            # Calculate context range
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            
            # Build context
            context_before = [lines[j].rstrip() for j in range(start, i)]
            context_after = [lines[j].rstrip() for j in range(i + 1, end)]
            
            results.append({
                "line_number": i + 1,
                "line_content": line.rstrip(),
                "context_before": context_before,
                "context_after": context_after,
            })
            
            if len(results) >= max_results:
                break
    
    return results


def _format_match(file_path: Path, match: dict, context_lines: int) -> str:
    """Format a single match with context for display."""
    lines = []
    line_num = match["line_number"]
    
    # Add context before
    for i, ctx_line in enumerate(match["context_before"]):
        ctx_num = line_num - len(match["context_before"]) + i
        lines.append(f"{ctx_num:4d} | {ctx_line}")
    
    # Add the match line
    lines.append(f"{line_num:4d} > {match['line_content']}")
    
    # Add context after
    for i, ctx_line in enumerate(match["context_after"]):
        ctx_num = line_num + i + 1
        lines.append(f"{ctx_num:4d} | {ctx_line}")
    
    return "\n".join(lines)


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = False,
    context_lines: int = 2,
    max_results: int = 50,
) -> str:
    """Execute search and return formatted results as string."""
    import json
    
    search_path = Path(path).expanduser().resolve()
    
    if not search_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Path does not exist: {path}",
            "matches": [],
        })
    
    matches = []
    total_matches = 0
    files_searched = 0
    
    try:
        if search_path.is_file():
            # Search in single file
            files_to_search = [search_path]
        else:
            # Search in directory
            if recursive:
                files_to_search = list(search_path.rglob("*"))
            else:
                files_to_search = list(search_path.iterdir())
            # Filter to files only
            files_to_search = [f for f in files_to_search if f.is_file()]
        
        # Apply file pattern filter
        if file_pattern:
            files_to_search = [
                f for f in files_to_search 
                if fnmatch.fnmatch(f.name, file_pattern)
            ]
        
        for file_path in files_to_search:
            # Skip binary files and common non-text files
            if file_path.suffix in {".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe"}:
                continue
            
            files_searched += 1
            file_matches = _read_file_with_context(
                file_path, pattern, case_sensitive, context_lines, max_results - total_matches
            )
            
            if file_matches:
                total_matches += len(file_matches)
                matches.append({
                    "file": str(file_path),
                    "matches": file_matches,
                })
                
                if total_matches >= max_results:
                    break
        
        # Build formatted output
        output_lines = [
            f"Search Results for '{pattern}':",
            f"Files searched: {files_searched}, Total matches: {total_matches}",
            "-" * 60,
        ]
        
        for file_match in matches:
            output_lines.append(f"\nFile: {file_match['file']}")
            output_lines.append("-" * 40)
            for match in file_match["matches"]:
                output_lines.append(_format_match(Path(file_match["file"]), match, context_lines))
                output_lines.append("")
        
        if total_matches >= max_results:
            output_lines.append("\n[Results truncated - max_results reached]")
        
        if total_matches == 0:
            output_lines.append("\nNo matches found.")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "matches": matches,
        })
