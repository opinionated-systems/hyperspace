"""
File info tool: get detailed metadata about files.

Provides file size, modification time, line count, and other metadata
to help the agent understand the codebase structure.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_info",
        "description": (
            "Get detailed metadata about a file or directory. "
            "Returns size, modification time, line count, and file type. "
            "Useful for understanding codebase structure before editing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory.",
                }
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict file info operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def tool_function(path: str) -> str:
    """Get detailed metadata about a file or directory."""
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

        stat = p.stat()
        
        # Basic metadata
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        if p.is_dir():
            # Directory info
            try:
                files = list(p.iterdir())
                file_count = sum(1 for f in files if f.is_file())
                dir_count = sum(1 for f in files if f.is_dir())
                return (
                    f"Directory: {path}\n"
                    f"  Size: {_format_size(size)}\n"
                    f"  Modified: {mtime}\n"
                    f"  Files: {file_count}\n"
                    f"  Subdirectories: {dir_count}\n"
                    f"  Total items: {len(files)}"
                )
            except Exception as e:
                return f"Error reading directory: {e}"
        else:
            # File info
            line_count = 0
            try:
                content = p.read_text(errors="ignore")
                line_count = len(content.splitlines())
            except Exception:
                pass
            
            file_type = "text"
            if p.suffix in [".py", ".pyw"]:
                file_type = "Python source"
            elif p.suffix in [".json", ".jsonl"]:
                file_type = "JSON data"
            elif p.suffix in [".md", ".rst", ".txt"]:
                file_type = "Documentation"
            elif p.suffix in [".yml", ".yaml"]:
                file_type = "YAML config"
            elif p.suffix in [".toml", ".cfg", ".ini"]:
                file_type = "Config file"
            
            return (
                f"File: {path}\n"
                f"  Type: {file_type}\n"
                f"  Size: {_format_size(size)} ({size} bytes)\n"
                f"  Lines: {line_count}\n"
                f"  Modified: {mtime}\n"
                f"  Extension: {p.suffix or 'none'}"
            )

    except Exception as e:
        return f"Error: {e}"
