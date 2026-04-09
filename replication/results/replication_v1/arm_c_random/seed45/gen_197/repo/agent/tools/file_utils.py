"""
File utility functions for common operations.

Provides helper functions for file operations that complement
the editor tool with higher-level operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


def find_files(
    root: str,
    pattern: str = "*",
    recursive: bool = True,
    exclude_dirs: list[str] | None = None,
) -> Iterator[Path]:
    """Find files matching a pattern.
    
    Args:
        root: Root directory to search from
        pattern: Glob pattern to match (default: "*" for all files)
        recursive: Whether to search recursively (default: True)
        exclude_dirs: Directory names to exclude (e.g., ["__pycache__", ".git"])
        
    Yields:
        Path objects for matching files
    """
    root_path = Path(root)
    exclude_dirs = exclude_dirs or []
    
    if recursive:
        for path in root_path.rglob(pattern):
            if path.is_file():
                # Check if any parent directory should be excluded
                if not any(part in exclude_dirs for part in path.parts):
                    yield path
    else:
        for path in root_path.glob(pattern):
            if path.is_file():
                yield path


def count_lines(path: str) -> dict:
    """Count lines in a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary with line counts:
        - total: total number of lines
        - code: lines of code (non-empty, non-comment)
        - blank: blank lines
        - comment: comment lines
    """
    p = Path(path)
    if not p.exists():
        return {"error": f"File not found: {path}"}
    
    content = p.read_text()
    lines = content.split("\n")
    
    total = len(lines)
    blank = sum(1 for line in lines if not line.strip())
    
    # Simple comment detection for Python files
    if path.endswith(".py"):
        comment = sum(1 for line in lines if line.strip().startswith("#"))
        code = total - blank - comment
    else:
        comment = 0
        code = total - blank
    
    return {
        "total": total,
        "code": code,
        "blank": blank,
        "comment": comment,
    }


def get_file_info(path: str) -> dict:
    """Get information about a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Dictionary with file information:
        - exists: whether the file exists
        - size: file size in bytes
        - modified: last modification time
        - extension: file extension
        - is_python: whether it's a Python file
    """
    p = Path(path)
    
    if not p.exists():
        return {"exists": False}
    
    stat = p.stat()
    extension = p.suffix
    
    return {
        "exists": True,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "extension": extension,
        "is_python": extension == ".py",
        "name": p.name,
        "parent": str(p.parent),
    }


def list_directory(
    path: str,
    show_hidden: bool = False,
    sort_by: str = "name",
) -> list[dict]:
    """List contents of a directory.
    
    Args:
        path: Path to the directory
        show_hidden: Whether to show hidden files (default: False)
        sort_by: Sort by "name", "size", or "modified" (default: "name")
        
    Returns:
        List of dictionaries with file/directory information
    """
    p = Path(path)
    if not p.is_dir():
        return [{"error": f"Not a directory: {path}"}]
    
    items = []
    for item in p.iterdir():
        if not show_hidden and item.name.startswith("."):
            continue
        
        stat = item.stat()
        items.append({
            "name": item.name,
            "path": str(item),
            "is_dir": item.is_dir(),
            "is_file": item.is_file(),
            "size": stat.st_size if item.is_file() else None,
            "modified": stat.st_mtime,
        })
    
    # Sort items
    if sort_by == "name":
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    elif sort_by == "size":
        items.sort(key=lambda x: x["size"] or 0, reverse=True)
    elif sort_by == "modified":
        items.sort(key=lambda x: x["modified"], reverse=True)
    
    return items


def search_in_files(
    root: str,
    query: str,
    file_pattern: str = "*.py",
    case_sensitive: bool = False,
) -> list[dict]:
    """Search for text in files.
    
    Args:
        root: Root directory to search from
        query: Text to search for
        file_pattern: Glob pattern for files to search (default: "*.py")
        case_sensitive: Whether search is case sensitive (default: False)
        
    Returns:
        List of dictionaries with search results:
        - file: path to the file
        - line: line number
        - content: line content
    """
    results = []
    root_path = Path(root)
    
    if not case_sensitive:
        query = query.lower()
    
    for file_path in root_path.rglob(file_pattern):
        if not file_path.is_file():
            continue
        
        try:
            content = file_path.read_text()
            lines = content.split("\n")
            
            for i, line in enumerate(lines, 1):
                check_line = line if case_sensitive else line.lower()
                if query in check_line:
                    results.append({
                        "file": str(file_path),
                        "line": i,
                        "content": line.strip(),
                    })
        except Exception:
            # Skip files that can't be read
            continue
    
    return results
