"""
File stats tool: get statistics about files (line count, word count, size).

Useful for the meta-agent to understand file sizes and complexity
when exploring the codebase structure.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_stats",
        "description": (
            "Get statistics about a file or directory. "
            "Returns line count, word count, character count, and file size. "
            "Useful for understanding file sizes and complexity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "For directories, include stats for all files recursively.",
                },
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _get_file_stats(file_path: Path) -> dict:
    """Get statistics for a single file."""
    try:
        content = file_path.read_text(errors="ignore")
        lines = content.split("\n")
        words = content.split()
        size = file_path.stat().st_size
        
        return {
            "path": str(file_path),
            "lines": len(lines),
            "words": len(words),
            "characters": len(content),
            "size_bytes": size,
            "size_human": _format_size(size),
        }
    except Exception as e:
        return {
            "path": str(file_path),
            "error": str(e),
        }


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def tool_function(path: str, recursive: bool = False) -> str:
    """Get statistics about a file or directory."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        if p.is_file():
            stats = _get_file_stats(p)
            if "error" in stats:
                return f"Error reading {path}: {stats['error']}"
            return _format_stats_output([stats])
        
        elif p.is_dir():
            if recursive:
                all_stats = []
                total_size = 0
                total_lines = 0
                total_words = 0
                
                for file_path in p.rglob("*"):
                    if file_path.is_file():
                        stats = _get_file_stats(file_path)
                        if "error" not in stats:
                            all_stats.append(stats)
                            total_size += stats["size_bytes"]
                            total_lines += stats["lines"]
                            total_words += stats["words"]
                
                # Sort by size (largest first)
                all_stats.sort(key=lambda x: x["size_bytes"], reverse=True)
                
                # Add summary
                summary = {
                    "path": f"{path} (total)",
                    "lines": total_lines,
                    "words": total_words,
                    "characters": sum(s["characters"] for s in all_stats),
                    "size_bytes": total_size,
                    "size_human": _format_size(total_size),
                }
                all_stats.insert(0, summary)
                
                return _format_stats_output(all_stats)
            else:
                # Just list files in directory with stats
                all_stats = []
                total_size = 0
                
                for file_path in p.iterdir():
                    if file_path.is_file():
                        stats = _get_file_stats(file_path)
                        if "error" not in stats:
                            all_stats.append(stats)
                            total_size += stats["size_bytes"]
                
                # Sort by size (largest first)
                all_stats.sort(key=lambda x: x["size_bytes"], reverse=True)
                
                return _format_stats_output(all_stats)
        
        else:
            return f"Error: {path} is neither a file nor a directory."
            
    except Exception as e:
        return f"Error: {e}"


def _format_stats_output(stats_list: list[dict]) -> str:
    """Format statistics as a readable table."""
    if not stats_list:
        return "No files found."
    
    # Calculate column widths
    path_width = max(len(s["path"]) for s in stats_list)
    path_width = min(path_width, 60)  # Cap at 60 chars
    
    lines = []
    lines.append("File Statistics:")
    lines.append("-" * (path_width + 50))
    lines.append(f"{'Path':<{path_width}} | {'Lines':>8} | {'Words':>8} | {'Size':>10}")
    lines.append("-" * (path_width + 50))
    
    for stats in stats_list:
        path = stats["path"]
        if len(path) > path_width:
            path = "..." + path[-(path_width-3):]
        lines.append(
            f"{path:<{path_width}} | {stats['lines']:>8} | {stats['words']:>8} | {stats['size_human']:>10}"
        )
    
    lines.append("-" * (path_width + 50))
    return "\n".join(lines)
