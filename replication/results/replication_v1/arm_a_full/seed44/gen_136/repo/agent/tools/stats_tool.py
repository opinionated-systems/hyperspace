"""
Stats tool: get file and directory statistics.

Provides useful statistics about files and directories to help
understand codebase structure before making modifications.
"""

from __future__ import annotations

import os
from pathlib import Path


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for the stats tool."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_root(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def tool_info() -> dict:
    return {
        "name": "stats",
        "description": (
            "Get statistics about files and directories. "
            "Useful for understanding codebase structure before making changes. "
            "Returns line counts, file sizes, and directory listings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file or directory to analyze.",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files (starting with .). Default: false.",
                },
            },
            "required": ["path"],
        },
    }


def _count_lines(filepath: str) -> int:
    """Count lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _get_file_stats(filepath: str) -> dict:
    """Get statistics for a single file."""
    try:
        stat = os.stat(filepath)
        return {
            "path": filepath,
            "size_bytes": stat.st_size,
            "lines": _count_lines(filepath),
        }
    except Exception as e:
        return {
            "path": filepath,
            "error": str(e),
        }


def _get_dir_stats(dirpath: str, include_hidden: bool = False) -> dict:
    """Get statistics for a directory."""
    try:
        entries = os.listdir(dirpath)
        if not include_hidden:
            entries = [e for e in entries if not e.startswith('.')]
        
        files = []
        dirs = []
        total_size = 0
        total_lines = 0
        errors = []
        
        for entry in entries:
            full_path = os.path.join(dirpath, entry)
            if os.path.isfile(full_path):
                file_stats = _get_file_stats(full_path)
                if "error" in file_stats:
                    errors.append(f"{entry}: {file_stats['error']}")
                else:
                    files.append(file_stats)
                    if "size_bytes" in file_stats:
                        total_size += file_stats["size_bytes"]
                    if "lines" in file_stats:
                        total_lines += file_stats["lines"]
            elif os.path.isdir(full_path):
                dirs.append({"name": entry, "path": full_path})
        
        # Sort files by lines (descending) to show largest files first
        files.sort(key=lambda x: x.get("lines", 0), reverse=True)
        
        result = {
            "path": dirpath,
            "total_files": len(files),
            "total_dirs": len(dirs),
            "total_size_bytes": total_size,
            "total_lines": total_lines,
            "files": files[:20],  # Limit to top 20 files by line count
            "subdirectories": dirs[:10],  # Limit to first 10 directories
        }
        
        if errors:
            result["errors"] = errors[:5]  # Limit error messages
        
        return result
    except PermissionError:
        return {
            "path": dirpath,
            "error": "Permission denied - cannot access directory",
        }
    except Exception as e:
        return {
            "path": dirpath,
            "error": str(e),
        }


def tool_function(path: str, include_hidden: bool = False) -> str:
    """Get statistics about a file or directory.
    
    Returns formatted statistics including line counts and file sizes.
    """
    if not _is_within_root(path):
        return f"Error: Path '{path}' is outside the allowed root."
    
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."
    
    if os.path.isfile(path):
        stats = _get_file_stats(path)
        if "error" in stats:
            return f"Error getting stats for '{path}': {stats['error']}"
        
        lines = stats.get("lines", 0)
        size = stats.get("size_bytes", 0)
        return f"File: {path}\n  Lines: {lines}\n  Size: {size} bytes"
    
    elif os.path.isdir(path):
        stats = _get_dir_stats(path, include_hidden)
        if "error" in stats:
            return f"Error getting stats for '{path}': {stats['error']}"
        
        result = [
            f"Directory: {path}",
            f"  Total files: {stats['total_files']}",
            f"  Total directories: {stats['total_dirs']}",
            f"  Total size: {stats['total_size_bytes']} bytes",
            f"  Total lines: {stats['total_lines']}",
            "",
            "Top files by line count:",
        ]
        
        for f in stats.get("files", []):
            name = os.path.basename(f["path"])
            lines = f.get("lines", 0)
            size = f.get("size_bytes", 0)
            result.append(f"  {name}: {lines} lines, {size} bytes")
        
        if stats.get("subdirectories"):
            result.append("")
            result.append("Subdirectories:")
            for d in stats["subdirectories"]:
                result.append(f"  {d['name']}/")
        
        # Display any errors encountered
        if stats.get("errors"):
            result.append("")
            result.append("Errors encountered:")
            for err in stats["errors"]:
                result.append(f"  - {err}")
        
        return "\n".join(result)
    
    return f"Error: '{path}' is neither a file nor a directory."
