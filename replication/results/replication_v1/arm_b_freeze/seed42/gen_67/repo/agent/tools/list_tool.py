"""
List tool for exploring directory structures.

Provides detailed directory listings with file metadata.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "list_directory",
        "description": (
            "List directory contents with detailed information. "
            "Shows files and subdirectories with sizes and modification times. "
            "Useful for exploring project structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (absolute)",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively (default: false)",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for recursive listing (default: 2)",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Whether to show hidden files (default: false)",
                },
            },
            "required": ["path"],
        },
    }


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _list_recursive(
    path: Path, 
    current_depth: int, 
    max_depth: int, 
    show_hidden: bool,
    prefix: str = ""
) -> list[str]:
    """Recursively list directory contents."""
    results = []
    
    try:
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return [f"{prefix}[Permission denied]"]
    except Exception as e:
        return [f"{prefix}[Error: {e}]"]
    
    for i, entry in enumerate(entries):
        if not show_hidden and entry.name.startswith('.'):
            continue
            
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        
        if entry.is_dir():
            results.append(f"{prefix}{connector}{entry.name}/")
            if current_depth < max_depth:
                extension = "    " if is_last else "│   "
                results.extend(_list_recursive(
                    entry, current_depth + 1, max_depth, show_hidden, prefix + extension
                ))
        else:
            try:
                stat = entry.stat()
                size = _format_size(stat.st_size)
                results.append(f"{prefix}{connector}{entry.name} ({size})")
            except Exception:
                results.append(f"{prefix}{connector}{entry.name}")
    
    return results


def tool_function(
    path: str,
    recursive: bool = False,
    max_depth: int = 2,
    show_hidden: bool = False,
) -> str:
    """List directory contents.

    Args:
        path: Directory path to list (absolute)
        recursive: Whether to list recursively
        max_depth: Maximum depth for recursive listing
        show_hidden: Whether to show hidden files

    Returns:
        String with directory listing
    """
    p = Path(path)
    
    if not p.is_absolute():
        return f"Error: {path} is not an absolute path."
    
    if not p.exists():
        return f"Error: {path} does not exist."
    
    if not p.is_dir():
        return f"Error: {path} is not a directory."
    
    if recursive:
        lines = [f"Directory tree of {path}:", ""]
        lines.extend(_list_recursive(p, 0, max_depth, show_hidden))
        return "\n".join(lines)
    else:
        # Simple listing
        try:
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return f"Error: Permission denied for {path}"
        except Exception as e:
            return f"Error: {e}"
        
        lines = [f"Contents of {path}:", ""]
        
        dirs = []
        files = []
        
        for entry in entries:
            if not show_hidden and entry.name.startswith('.'):
                continue
                
            if entry.is_dir():
                dirs.append(f"  📁 {entry.name}/")
            else:
                try:
                    stat = entry.stat()
                    size = _format_size(stat.st_size)
                    files.append(f"  📄 {entry.name} ({size})")
                except Exception:
                    files.append(f"  📄 {entry.name}")
        
        lines.extend(dirs)
        lines.extend(files)
        
        lines.append("")
        lines.append(f"Total: {len(dirs)} directories, {len(files)} files")
        
        return "\n".join(lines)
