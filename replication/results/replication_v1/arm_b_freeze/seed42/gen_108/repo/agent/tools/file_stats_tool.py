"""
File statistics tool: get detailed stats about files and directories.

Provides information like file size, line counts, modification times,
and directory summaries. Useful for understanding codebase structure.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "file_stats",
        "description": (
            "Get detailed statistics about files and directories. "
            "Provides file size, line count, modification time, and directory summaries. "
            "Useful for understanding codebase structure and identifying large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory.",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files in directory stats (default: false).",
                },
            },
            "required": ["path"],
        },
    }


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_file_stats(filepath: str) -> dict[str, Any]:
    """Get statistics for a single file."""
    try:
        stat = os.stat(filepath)
        size = stat.st_size
        
        # Count lines if it's a text file
        line_count = None
        if filepath.endswith(('.py', '.txt', '.md', '.json', '.yaml', '.yml', '.js', '.ts', '.html', '.css', '.sh')):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                pass
        
        return {
            "type": "file",
            "size_bytes": size,
            "size_human": _format_size(size),
            "line_count": line_count,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }
    except Exception as e:
        return {"type": "file", "error": str(e)}


def _get_dir_stats(dirpath: str, include_hidden: bool = False) -> dict[str, Any]:
    """Get statistics for a directory."""
    try:
        total_size = 0
        file_count = 0
        dir_count = 0
        extension_counts: dict[str, int] = {}
        largest_files: list[tuple[str, int]] = []
        
        for root, dirs, files in os.walk(dirpath):
            # Filter hidden directories
            if not include_hidden:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for d in dirs:
                dir_count += 1
            
            for f in files:
                if not include_hidden and f.startswith('.'):
                    continue
                
                filepath = os.path.join(root, f)
                try:
                    size = os.path.getsize(filepath)
                    total_size += size
                    file_count += 1
                    
                    # Track extensions
                    ext = os.path.splitext(f)[1].lower() or "(no ext)"
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
                    
                    # Track largest files (keep top 5)
                    largest_files.append((filepath, size))
                    largest_files.sort(key=lambda x: x[1], reverse=True)
                    largest_files = largest_files[:5]
                    
                except Exception:
                    pass
        
        # Format largest files
        largest_formatted = [
            f"  {_format_size(size)}: {path.replace(dirpath, '.')}"
            for path, size in largest_files
        ]
        
        # Sort extensions by count
        sorted_extensions = sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "type": "directory",
            "file_count": file_count,
            "dir_count": dir_count,
            "total_size_bytes": total_size,
            "total_size_human": _format_size(total_size),
            "top_extensions": dict(sorted_extensions),
            "largest_files": largest_formatted,
        }
    except Exception as e:
        return {"type": "directory", "error": str(e)}


def tool_function(path: str, include_hidden: bool = False) -> str:
    """Get statistics about a file or directory.

    Args:
        path: Absolute path to file or directory.
        include_hidden: Whether to include hidden files in directory stats.

    Returns:
        Formatted string with file/directory statistics.
    """
    p = Path(path)
    
    if not p.is_absolute():
        return f"Error: {path} is not an absolute path."
    
    if not p.exists():
        return f"Error: {path} does not exist."
    
    if p.is_file():
        stats = _get_file_stats(str(p))
        lines = [
            f"File: {path}",
            f"  Size: {stats.get('size_human', 'N/A')} ({stats.get('size_bytes', 0):,} bytes)",
        ]
        if stats.get('line_count') is not None:
            lines.append(f"  Lines: {stats['line_count']:,}")
        lines.extend([
            f"  Modified: {stats.get('modified', 'N/A')}",
            f"  Created: {stats.get('created', 'N/A')}",
        ])
        return "\n".join(lines)
    
    else:
        stats = _get_dir_stats(str(p), include_hidden)
        lines = [
            f"Directory: {path}",
            f"  Files: {stats.get('file_count', 0):,}",
            f"  Subdirectories: {stats.get('dir_count', 0):,}",
            f"  Total size: {stats.get('total_size_human', 'N/A')} ({stats.get('total_size_bytes', 0):,} bytes)",
            "",
            "Top file extensions:",
        ]
        for ext, count in stats.get('top_extensions', {}).items():
            lines.append(f"  {ext}: {count}")
        
        if stats.get('largest_files'):
            lines.extend(["", "Largest files:"])
            lines.extend(stats['largest_files'])
        
        return "\n".join(lines)
