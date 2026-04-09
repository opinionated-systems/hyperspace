"""
List directory tool: list files and directories with optional recursion.

Provides a safe way to explore directory structures without using bash commands.
Includes options for filtering, recursion depth, and showing file sizes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "list_directory",
        "description": (
            "List files and directories at a given path. "
            "Supports recursive listing with depth limit, file filtering, "
            "and showing file sizes. Safer than using bash for directory exploration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list. Defaults to current working directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively. Default: false.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum recursion depth (1-10). Default: 3. Only used if recursive is true.",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Whether to show hidden files (starting with .). Default: false.",
                },
                "show_size": {
                    "type": "boolean",
                    "description": "Whether to show file sizes. Default: true.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py', '*.txt'). Default: no filter.",
                },
            },
            "required": [],
        },
    }


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _list_item(path: Path, show_size: bool, prefix: str = "") -> str:
    """Format a single directory item."""
    if path.is_dir():
        return f"{prefix}[DIR]  {path.name}/"
    else:
        size_str = ""
        if show_size:
            try:
                size = path.stat().st_size
                size_str = f" ({_format_size(size)})"
            except OSError:
                size_str = " (size unknown)"
        return f"{prefix}[FILE] {path.name}{size_str}"


def _list_recursive(
    base_path: Path,
    current_depth: int,
    max_depth: int,
    show_hidden: bool,
    show_size: bool,
    pattern: str | None,
    prefix: str = "",
) -> list[str]:
    """Recursively list directory contents."""
    results = []
    
    try:
        items = sorted(base_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return [f"{prefix}[PERMISSION DENIED]"]
    except OSError as e:
        return [f"{prefix}[ERROR: {e}]"]
    
    for item in items:
        # Skip hidden files unless show_hidden is True
        if not show_hidden and item.name.startswith("."):
            continue
        
        # Apply pattern filter (only for files)
        if pattern and item.is_file():
            if not item.match(pattern):
                continue
        
        results.append(_list_item(item, show_size, prefix))
        
        # Recurse into subdirectories if within max_depth
        if item.is_dir() and current_depth < max_depth:
            sub_results = _list_recursive(
                item,
                current_depth + 1,
                max_depth,
                show_hidden,
                show_size,
                pattern,
                prefix + "  ",
            )
            results.extend(sub_results)
    
    return results


def tool_function(
    path: str = ".",
    recursive: bool = False,
    max_depth: int = 3,
    show_hidden: bool = False,
    show_size: bool = True,
    pattern: str | None = None,
) -> str:
    """List directory contents with optional filtering and recursion.
    
    Args:
        path: Directory path to list
        recursive: Whether to list recursively
        max_depth: Maximum recursion depth (1-10, default 3)
        show_hidden: Whether to show hidden files
        show_size: Whether to show file sizes
        pattern: Optional glob pattern to filter files
        
    Returns:
        Formatted string listing directory contents
    """
    # Validate and normalize path
    target_path = Path(path).expanduser().resolve()
    
    if not target_path.exists():
        return f"Error: Path does not exist: {path}"
    
    if not target_path.is_dir():
        return f"Error: Path is not a directory: {path}"
    
    # Clamp max_depth to valid range
    max_depth = max(1, min(10, max_depth))
    
    # Build output
    lines = [f"Contents of: {target_path}", "=" * 50]
    
    if recursive:
        items = _list_recursive(
            target_path,
            current_depth=1,
            max_depth=max_depth,
            show_hidden=show_hidden,
            show_size=show_size,
            pattern=pattern,
        )
        lines.extend(items)
    else:
        try:
            dir_items = sorted(target_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for item in dir_items:
                if not show_hidden and item.name.startswith("."):
                    continue
                if pattern and item.is_file() and not item.match(pattern):
                    continue
                lines.append(_list_item(item, show_size))
        except PermissionError:
            lines.append("[PERMISSION DENIED]")
        except OSError as e:
            lines.append(f"[ERROR: {e}]")
    
    lines.append("=" * 50)
    return "\n".join(lines)
