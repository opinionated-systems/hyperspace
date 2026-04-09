"""
View tree tool: display directory structure in a tree-like format.

Provides a visual representation of the directory hierarchy,
useful for understanding codebase structure at a glance.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "view_tree",
        "description": (
            "Display directory structure in a tree-like format. "
            "Shows files and subdirectories with indentation to visualize hierarchy. "
            "Useful for understanding codebase structure at a glance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to display (default: current directory)",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth to display (default: 3)",
                },
                "show_files": {
                    "type": "boolean",
                    "description": "Whether to show files (default: True)",
                },
                "show_size": {
                    "type": "boolean",
                    "description": "Whether to show file sizes (default: False)",
                },
            },
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for tree viewing."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_root(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    try:
        Path(path).resolve().relative_to(Path(_ALLOWED_ROOT).resolve())
        return True
    except ValueError:
        return False


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}K"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}M"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}G"


def _build_tree(
    directory: Path,
    prefix: str = "",
    max_depth: int = 3,
    current_depth: int = 0,
    show_files: bool = True,
    show_size: bool = False,
) -> list[str]:
    """Recursively build tree structure lines."""
    if current_depth >= max_depth:
        return []

    lines = []
    
    try:
        entries = sorted(
            directory.iterdir(),
            key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except PermissionError:
        return [f"{prefix}[permission denied]"]
    except OSError as e:
        return [f"{prefix}[error: {e}]"]

    # Filter out unwanted entries
    filtered = [
        e for e in entries
        if not e.name.startswith(".")  # Hidden files
        and e.name != "__pycache__"
        and e.name != ".git"
        and e.name != "node_modules"
        and not e.name.endswith(".pyc")
    ]

    for i, entry in enumerate(filtered):
        is_last = i == len(filtered) - 1
        connector = "└── " if is_last else "├── "
        
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            extension = "    " if is_last else "│   "
            lines.extend(
                _build_tree(
                    entry,
                    prefix + extension,
                    max_depth,
                    current_depth + 1,
                    show_files,
                    show_size,
                )
            )
        elif show_files:
            name = entry.name
            if show_size and entry.is_file():
                try:
                    size = entry.stat().st_size
                    name = f"{name} ({_format_size(size)})"
                except OSError:
                    pass
            lines.append(f"{prefix}{connector}{name}")

    return lines


def tool_function(
    path: str = ".",
    max_depth: int = 3,
    show_files: bool = True,
    show_size: bool = False,
) -> str:
    """Display directory structure in a tree-like format.

    Args:
        path: Directory path to display
        max_depth: Maximum depth to display (1-10)
        show_files: Whether to show files (False shows only directories)
        show_size: Whether to show file sizes

    Returns:
        Tree-formatted string of the directory structure
    """
    # Validate and sanitize inputs
    max_depth = max(1, min(10, max_depth))
    
    target_path = Path(path or _ALLOWED_ROOT or ".").resolve()
    
    if not _is_within_root(str(target_path)):
        return f"Error: Path '{path}' is outside allowed root."

    if not target_path.exists():
        return f"Error: Path '{path}' does not exist."

    if not target_path.is_dir():
        return f"Error: Path '{path}' is not a directory."

    try:
        # Build tree
        tree_lines = [str(target_path) + "/"]
        tree_lines.extend(_build_tree(target_path, "", max_depth, 0, show_files, show_size))
        
        # Add summary
        file_count = sum(1 for _ in target_path.rglob("*") if _.is_file())
        dir_count = sum(1 for _ in target_path.rglob("*") if _.is_dir())
        
        tree_lines.append("")
        tree_lines.append(f"Summary: {dir_count} directories, {file_count} files")
        
        return "\n".join(tree_lines)

    except Exception as e:
        return f"Error building tree: {e}"
