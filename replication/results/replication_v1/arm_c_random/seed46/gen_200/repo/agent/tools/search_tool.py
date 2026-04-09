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
            "Commands: grep (search for text pattern), find (find files by name), count (count pattern occurrences). "
            "Useful for locating code that needs modification. "
            "Searches are case-insensitive and exclude hidden files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find", "count"],
                    "description": "The search command to run. 'grep' searches file contents, 'find' searches filenames, 'count' counts occurrences.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (grep) or filename pattern (find). Supports partial matches.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root). Must be absolute path.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js'). Include the dot.",
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
            return _grep(pattern, search_root, file_extension)
        elif command == "find":
            return _find(pattern, search_root, file_extension)
        elif command == "count":
            return _count(pattern, search_root, file_extension)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, search_root: str, file_extension: str | None) -> str:
    """Search for pattern in files using grep with ripgrep if available."""
    # Try ripgrep first (faster)
    try:
        rg_cmd = ["rg", "-n", "-i", "--no-heading", "--with-filename", pattern, search_root]
        if file_extension:
            rg_cmd.extend(["-g", f"*{file_extension}"])
        # Exclude hidden directories and __pycache__
        rg_cmd.extend(["-g", "!*/.*", "-g", "!*/__pycache__/*"])
        
        result = subprocess.run(
            rg_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _truncate(result.stdout, 8000)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Fall back to find + grep
    
    # Fallback to find + grep
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
        
        # Run grep on files
        matches = []
        for f in files[:100]:  # Limit to first 100 files for performance
            if not f:
                continue
            try:
                grep_result = subprocess.run(
                    ["grep", "-n", "-H", "-i", pattern, f],
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
    """Find files by name pattern using fd if available."""
    # Try fd first (faster)
    try:
        fd_cmd = ["fd", "-t", "f", "-a", pattern, search_root]
        if file_extension:
            fd_cmd.extend(["-e", file_extension.lstrip(".")])
        # Exclude hidden directories and __pycache__
        fd_cmd.extend(["-E", ".*", "-E", "__pycache__"])
        
        result = subprocess.run(
            fd_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            files = result.stdout.strip().split("\n")
            return f"Found {len(files)} file(s):\n" + _truncate(result.stdout, 5000)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Fall back to find
    
    # Fallback to find
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


def _count(pattern: str, search_root: str, file_extension: str | None) -> str:
    """Count occurrences of a pattern in files."""
    # Get list of files to search
    find_cmd = ["find", search_root, "-type", "f"]
    
    if file_extension:
        find_cmd.extend(["-name", f"*{file_extension}"])
    
    # Exclude hidden directories and __pycache__
    find_cmd.extend(["-not", "-path", "*/\.*", "-not", "-path", "*/__pycache__/*"])
    
    try:
        result = subprocess.run(
            find_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        files = [f for f in files if f]  # Remove empty strings
        
        if not files:
            return f"No files found in {search_root}"
        
        # Count occurrences in each file
        total_count = 0
        file_counts = []
        
        for f in files[:200]:  # Limit to first 200 files for performance
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    count = content.lower().count(pattern.lower())
                    if count > 0:
                        total_count += count
                        # Get relative path for cleaner output
                        rel_path = f.replace(search_root, "").lstrip("/")
                        file_counts.append((rel_path, count))
            except Exception:
                continue
        
        if total_count == 0:
            return f"No occurrences of '{pattern}' found"
        
        # Sort by count descending
        file_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Build result
        lines = [f"Total occurrences of '{pattern}': {total_count}", ""]
        lines.append("Top files by occurrence count:")
        for rel_path, count in file_counts[:20]:  # Show top 20
            lines.append(f"  {count:4d}  {rel_path}")
        
        if len(file_counts) > 20:
            lines.append(f"  ... and {len(file_counts) - 20} more files")
        
        return "\n".join(lines)
        
    except subprocess.TimeoutExpired:
        return "Error: count operation timed out"
    except Exception as e:
        return f"Error during count: {e}"
