"""
File stats tool: analyze file statistics and codebase metrics.

Provides insights into file sizes, line counts, and codebase structure
to help the agent understand the scope and complexity of files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _count_lines(filepath: str) -> int:
    """Count the number of lines in a file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(filepath)
    except Exception:
        return 0


def analyze_file(filepath: str) -> dict[str, Any]:
    """Analyze a single file and return statistics."""
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    if not path.is_file():
        return {"error": f"Not a file: {filepath}"}
    
    stats = {
        "path": str(path.absolute()),
        "filename": path.name,
        "extension": path.suffix,
        "size_bytes": _get_file_size(filepath),
        "line_count": _count_lines(filepath),
        "is_python": path.suffix == ".py",
        "is_text": path.suffix in [".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml"],
    }
    return stats


def analyze_directory(dirpath: str, recursive: bool = False) -> dict[str, Any]:
    """Analyze a directory and return aggregate statistics."""
    path = Path(dirpath)
    if not path.exists():
        return {"error": f"Directory not found: {dirpath}"}
    
    if not path.is_dir():
        return {"error": f"Not a directory: {dirpath}"}
    
    files = []
    total_size = 0
    total_lines = 0
    python_files = 0
    
    if recursive:
        file_iter = path.rglob("*")
    else:
        file_iter = path.iterdir()
    
    for item in file_iter:
        if item.is_file():
            file_info = analyze_file(str(item))
            if "error" not in file_info:
                files.append(file_info)
                total_size += file_info["size_bytes"]
                total_lines += file_info["line_count"]
                if file_info["is_python"]:
                    python_files += 1
    
    return {
        "path": str(path.absolute()),
        "total_files": len(files),
        "python_files": python_files,
        "total_size_bytes": total_size,
        "total_lines": total_lines,
        "files": files[:20],  # Limit to first 20 files
        "truncated": len(files) > 20,
    }


def tool_info() -> dict[str, Any]:
    return {
        "name": "file_stats",
        "description": "Analyze file statistics including size, line count, and codebase metrics. Helps understand the scope and complexity of files before editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to analyze",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "For directories: analyze recursively including subdirectories",
                    "default": False,
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str, recursive: bool = False) -> str:
    """Analyze file or directory statistics."""
    import json
    
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"Path not found: {path}"}, indent=2)
    
    if p.is_file():
        result = analyze_file(path)
    else:
        result = analyze_directory(path, recursive)
    
    return json.dumps(result, indent=2, default=str)
