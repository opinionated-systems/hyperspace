"""
File search tool: search for files by name pattern or content.

Provides capabilities to:
1. Search for files by name pattern (glob)
2. Search for files containing specific text
3. Search for files by extension
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool specification for LLM function calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": (
                "Search for files in a directory by name pattern, content, or extension. "
                "Returns a list of matching file paths with optional preview of matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in. Defaults to current directory.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern for file names (e.g., '*.py', 'test_*.py').",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to search for within files.",
                    },
                    "extension": {
                        "type": "string",
                        "description": "File extension to filter by (e.g., 'py', 'txt').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default 50.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively. Default True.",
                    },
                },
            },
            "required": [],
        },
    }


def _search_by_pattern(
    base_path: Path,
    pattern: str | None,
    extension: str | None,
    recursive: bool,
    max_results: int,
) -> list[Path]:
    """Search for files matching name pattern and/or extension."""
    matches = []
    
    if recursive:
        iterator = base_path.rglob("*")
    else:
        iterator = base_path.iterdir()
    
    for item in iterator:
        if not item.is_file():
            continue
        
        # Check extension filter
        if extension is not None:
            if item.suffix.lstrip(".") != extension:
                continue
        
        # Check pattern filter
        if pattern is not None:
            if not fnmatch.fnmatch(item.name, pattern):
                continue
        
        matches.append(item)
        if len(matches) >= max_results:
            break
    
    return matches


def _search_by_content(
    base_path: Path,
    content: str,
    pattern: str | None,
    extension: str | None,
    recursive: bool,
    max_results: int,
) -> list[tuple[Path, list[str]]]:
    """Search for files containing specific text."""
    matches = []
    search_regex = re.compile(re.escape(content), re.IGNORECASE)
    
    # First get candidate files by pattern/extension if specified
    if pattern is not None or extension is not None:
        candidates = _search_by_pattern(base_path, pattern, extension, recursive, 10000)
    else:
        if recursive:
            candidates = [p for p in base_path.rglob("*") if p.is_file()]
        else:
            candidates = [p for p in base_path.iterdir() if p.is_file()]
    
    for file_path in candidates:
        try:
            # Skip binary files
            if _is_binary(file_path):
                continue
            
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            matching_lines = []
            for i, line in enumerate(lines, 1):
                if search_regex.search(line):
                    matching_lines.append(f"Line {i}: {line.rstrip()}")
                    if len(matching_lines) >= 3:  # Limit preview lines per file
                        break
            
            if matching_lines:
                matches.append((file_path, matching_lines))
                if len(matches) >= max_results:
                    break
        except Exception as e:
            logger.debug("Error reading %s: %s", file_path, e)
            continue
    
    return matches


def _is_binary(file_path: Path) -> bool:
    """Check if a file is binary by reading first chunk."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
        return b"\0" in chunk
    except Exception:
        return True


def tool_function(
    path: str = ".",
    pattern: str | None = None,
    content: str | None = None,
    extension: str | None = None,
    max_results: int = 50,
    recursive: bool = True,
    **kwargs: Any,
) -> str:
    """Execute file search and return results."""
    base_path = Path(path).expanduser().resolve()
    
    if not base_path.exists():
        return f"Error: Path '{path}' does not exist."
    if not base_path.is_dir():
        return f"Error: Path '{path}' is not a directory."
    
    # Validate max_results
    max_results = min(max(1, max_results), 100)
    
    results = []
    
    if content is not None:
        # Content search
        content_matches = _search_by_content(
            base_path, content, pattern, extension, recursive, max_results
        )
        
        if not content_matches:
            results.append(f"No files found containing '{content}'")
        else:
            results.append(f"Found {len(content_matches)} file(s) containing '{content}':")
            for file_path, lines in content_matches:
                rel_path = file_path.relative_to(base_path)
                results.append(f"\n{rel_path}:")
                for line in lines:
                    results.append(f"  {line}")
    else:
        # Pattern/extension search only
        file_matches = _search_by_pattern(
            base_path, pattern, extension, recursive, max_results
        )
        
        if not file_matches:
            filter_desc = []
            if pattern:
                filter_desc.append(f"pattern '{pattern}'")
            if extension:
                filter_desc.append(f"extension '.{extension}'")
            desc = " and ".join(filter_desc) if filter_desc else "any files"
            results.append(f"No files found matching {desc}")
        else:
            filter_desc = []
            if pattern:
                filter_desc.append(f"pattern '{pattern}'")
            if extension:
                filter_desc.append(f"extension '.{extension}'")
            desc = " and ".join(filter_desc) if filter_desc else "all files"
            results.append(f"Found {len(file_matches)} file(s) matching {desc}:")
            for file_path in file_matches:
                rel_path = file_path.relative_to(base_path)
                results.append(f"  {rel_path}")
    
    return "\n".join(results)
