"""
File info tool: get detailed metadata about files and directories.

Provides file size, modification time, permissions, and other metadata.
Useful for exploring the codebase structure.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_info",
        "description": (
            "Get detailed metadata about a file or directory. "
            "Returns size, modification time, permissions, and type information. "
            "Useful for exploring codebase structure and checking file properties."
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
                    "description": "Include hidden files when listing directories (default: False).",
                },
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
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _format_permissions(mode: int) -> str:
    """Format file permissions in Unix-style rwx format."""
    perms = ""
    perms += "r" if mode & stat.S_IRUSR else "-"
    perms += "w" if mode & stat.S_IWUSR else "-"
    perms += "x" if mode & stat.S_IXUSR else "-"
    perms += "r" if mode & stat.S_IRGRP else "-"
    perms += "w" if mode & stat.S_IWGRP else "-"
    perms += "x" if mode & stat.S_IXGRP else "-"
    perms += "r" if mode & stat.S_IROTH else "-"
    perms += "w" if mode & stat.S_IWOTH else "-"
    perms += "x" if mode & stat.S_IXOTH else "-"
    return perms


def tool_function(
    path: str,
    include_hidden: bool = False,
) -> str:
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
            return f"Error: {p} does not exist."

        # Get file stats
        try:
            stat_info = p.stat()
        except (OSError, IOError) as e:
            return f"Error: Cannot access {p}: {e}"

        # Build metadata info
        lines = [f"File: {p}", f"Type: {'Directory' if p.is_dir() else 'File'}"]

        # Size
        if p.is_file():
            lines.append(f"Size: {_format_size(stat_info.st_size)} ({stat_info.st_size} bytes)")
        elif p.is_dir():
            try:
                entries = list(p.iterdir())
                visible = [e for e in entries if not e.name.startswith(".")]
                hidden = [e for e in entries if e.name.startswith(".")]
                lines.append(f"Entries: {len(visible)} visible, {len(hidden)} hidden")
            except (OSError, PermissionError):
                lines.append("Entries: unable to count")

        # Timestamps
        mtime = datetime.fromtimestamp(stat_info.st_mtime)
        lines.append(f"Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        ctime = datetime.fromtimestamp(stat_info.st_ctime)
        lines.append(f"Created:  {ctime.strftime('%Y-%m-%d %H:%M:%S')}")

        # Permissions
        mode = stat_info.st_mode
        lines.append(f"Permissions: {_format_permissions(mode)} ({oct(mode)[-3:]})")

        # Owner/Group (if available)
        try:
            import pwd
            import grp
            owner = pwd.getpwuid(stat_info.st_uid).pw_name
            group = grp.getgrgid(stat_info.st_gid).gr_name
            lines.append(f"Owner: {owner}:{group}")
        except (ImportError, KeyError):
            lines.append(f"UID: {stat_info.st_uid}, GID: {stat_info.st_gid}")

        # For directories, optionally list contents
        if p.is_dir():
            lines.append("")
            lines.append("Contents:")
            try:
                entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
                count = 0
                for entry in entries:
                    if not include_hidden and entry.name.startswith("."):
                        continue
                    count += 1
                    if count > 50:
                        lines.append(f"  ... ({len(entries) - 50} more entries)")
                        break
                    
                    prefix = "📁 " if entry.is_dir() else "📄 "
                    try:
                        entry_stat = entry.stat()
                        size_str = _format_size(entry_stat.st_size) if entry.is_file() else ""
                        lines.append(f"  {prefix}{entry.name:<30} {size_str}")
                    except (OSError, IOError):
                        lines.append(f"  {prefix}{entry.name:<30} [inaccessible]")
            except (OSError, PermissionError) as e:
                lines.append(f"  [Unable to list: {e}]")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"
