"""
View tool: view directory structures and file listings.

Provides tree-like directory viewing and file listing capabilities
to help the agent navigate and understand codebase structure.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "view",
        "description": (
            "View tool for exploring directory structures and file listings. "
            "Commands: view_directory (tree-like listing), view_file_list (simple file list)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view_directory", "view_file_list"],
                    "description": "The view command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to view (absolute path).",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for directory tree (default 3).",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Whether to show hidden files (default false).",
                },
            },
            "required": ["command", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict view operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    command: str,
    path: str,
    max_depth: int = 3,
    show_hidden: bool = False,
) -> str:
    """Execute a view command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {path} does not exist."

        if not p.is_dir():
            return f"Error: {path} is not a directory."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if command == "view_directory":
            return _view_directory(p, max_depth, show_hidden)
        elif command == "view_file_list":
            return _view_file_list(p, show_hidden)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _view_directory(base: Path, max_depth: int, show_hidden: bool) -> str:
    """Generate a tree-like view of the directory structure."""
    lines = [f"Directory tree of {base}:", ""]
    
    def _tree(path: Path, prefix: str = "", depth: int = 0) -> list[str]:
        if depth > max_depth:
            return []
        
        result = []
        try:
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return [f"{prefix}[permission denied]"]
        
        # Filter hidden files
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            
            if entry.is_dir():
                result.append(f"{prefix}{connector}{entry.name}/")
                if depth < max_depth:
                    extension = "    " if is_last else "│   "
                    result.extend(_tree(entry, prefix + extension, depth + 1))
            else:
                result.append(f"{prefix}{connector}{entry.name}")
        
        return result
    
    lines.extend(_tree(base, "", 0))
    return "\n".join(lines)


def _view_file_list(base: Path, show_hidden: bool) -> str:
    """Generate a simple flat file list."""
    try:
        if show_hidden:
            files = list(base.rglob("*"))
        else:
            files = [f for f in base.rglob("*") if not any(p.startswith(".") for p in f.relative_to(base).parts)]
        
        files = sorted([f for f in files if f.is_file()])
        
        if not files:
            return f"No files found in {base}"
        
        lines = [f"Files in {base}:", ""]
        for f in files:
            rel_path = f.relative_to(base)
            lines.append(str(rel_path))
        
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing files: {e}"
