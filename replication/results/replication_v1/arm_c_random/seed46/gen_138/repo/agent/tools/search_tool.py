"""
Search tool: find files and search for patterns in the codebase.

Provides grep-like functionality and file finding capabilities
to help the meta agent locate code that needs modification.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and patterns in the codebase. "
            "Commands: grep (search for text pattern), find (find files by name), "
            "preview (show content preview of a file). "
            "Useful for locating code that needs modification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find", "preview"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (grep) or filename pattern (find) or file path (preview).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines for grep (default: 2).",
                },
                "max_preview_lines": {
                    "type": "integer",
                    "description": "Maximum lines to show in preview (default: 50).",
                },
            },
            "required": ["command", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate(content: str, max_len: int = 5000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    context_lines: int = 2,
    max_preview_lines: int = 50,
) -> str:
    """Execute a search command."""
    try:
        # Determine search root
        search_root = path or _ALLOWED_ROOT or os.getcwd()
        search_root = os.path.abspath(search_root)

        # Scope check
        if _ALLOWED_ROOT is not None:
            if not search_root.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if command == "grep":
            return _grep(pattern, search_root, file_extension, context_lines)
        elif command == "find":
            return _find(pattern, search_root, file_extension)
        elif command == "preview":
            return _preview(pattern, max_preview_lines)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, search_root: str, file_extension: str | None, context_lines: int = 2) -> str:
    """Search for pattern in files using grep with context lines."""
    # Build find command to get files
    find_cmd = ["find", search_root, "-type", "f"]
    
    if file_extension:
        find_cmd.extend(["-name", f"*{file_extension}"])
    
    # Exclude hidden directories and __pycache__
    find_cmd.extend(["-not", "-path", "*/\.*", "-not", "-path", "*/__pycache__/*"])
    
    try:
        # Get list of files
        result = subprocess.run(
            find_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        
        if not files or files == ['']:
            return f"No files found in {search_root}"
        
        # Run grep on files with context lines
        matches = []
        grep_args = ["grep", "-n", "-H", "-i"]
        if context_lines > 0:
            grep_args.extend(["-C", str(context_lines)])
        grep_args.extend([pattern])
        
        for f in files[:100]:  # Limit to first 100 files for performance
            if not f:
                continue
            try:
                file_grep_args = grep_args + [f]
                grep_result = subprocess.run(
                    file_grep_args,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if grep_result.returncode == 0 and grep_result.stdout:
                    matches.append(grep_result.stdout)
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue
        
        if matches:
            output = "".join(matches)
            return _truncate(output, 8000)
        else:
            return f"No matches found for '{pattern}'"
            
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except Exception as e:
        return f"Error during search: {e}"


def _find(pattern: str, search_root: str, file_extension: str | None) -> str:
    """Find files by name pattern."""
    find_cmd = ["find", search_root, "-type", "f", "-name", f"*{pattern}*"]
    
    if file_extension:
        find_cmd[-1] = f"*{pattern}*{file_extension}"
    
    # Exclude hidden directories and __pycache__
    find_cmd.extend(["-not", "-path", "*/\.*", "-not", "-path", "*/__pycache__/*"])
    
    try:
        result = subprocess.run(
            find_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            files = result.stdout.strip().split("\n")
            return f"Found {len(files)} file(s):\n" + _truncate(result.stdout, 5000)
        else:
            return f"No files found matching '{pattern}'"
            
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except Exception as e:
        return f"Error during find: {e}"


def _preview(file_path: str, max_lines: int = 50) -> str:
    """Show a preview of a file's content with line numbers."""
    try:
        p = Path(file_path)
        
        if not p.is_absolute():
            return f"Error: {file_path} is not an absolute path."
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if p.is_dir():
            return f"Error: {p} is a directory, not a file."
        
        # Read file content
        content = p.read_text()
        lines = content.split("\n")
        total_lines = len(lines)
        
        # Determine how many lines to show
        show_lines = min(max_lines, total_lines)
        
        # Format with line numbers
        numbered_lines = []
        for i, line in enumerate(lines[:show_lines], 1):
            numbered_lines.append(f"{i:4d} | {line}")
        
        preview = "\n".join(numbered_lines)
        
        if total_lines > show_lines:
            preview += f"\n... ({total_lines - show_lines} more lines)"
        
        header = f"Preview of {p} ({total_lines} total lines, showing first {show_lines}):\n"
        return header + preview
        
    except Exception as e:
        return f"Error previewing file: {e}"
