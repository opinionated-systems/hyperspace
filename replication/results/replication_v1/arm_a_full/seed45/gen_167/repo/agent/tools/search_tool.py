"""
File search tool: search for files by name or content patterns.

Provides grep-like functionality to find files containing specific text,
or find files matching name patterns. Useful for exploring large codebases.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files by content pattern or filename pattern. "
            "Uses grep for content search and find for filename search. "
            "Results are truncated if too large."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (text to find in files, or glob pattern for filenames).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Type of search: 'content' searches file contents, 'filename' searches filenames.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt'). Only for content search.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50, max: 200).",
                },
            },
            "required": ["pattern", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate(content: str, max_len: int = 15000) -> str:
    """Truncate content if it exceeds max length."""
    if len(content) > max_len:
        lines = content.split("\n")
        # Keep first half and last half of lines
        half_count = len(lines) // 2
        kept_lines = lines[:half_count] + ["... [output truncated, too many results] ..."] + lines[-half_count:]
        return "\n".join(kept_lines)
    return content


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int | None = None,
) -> str:
    """Execute a file search.
    
    Args:
        pattern: Text pattern to search for (content) or glob pattern (filename)
        search_type: Either 'content' or 'filename'
        path: Directory to search in (default: current directory or allowed root)
        file_extension: Optional extension filter for content search (e.g., '.py')
        max_results: Maximum results to return (default: 50, max: 200)
    """
    # Validate inputs
    if not pattern or not pattern.strip():
        return "Error: pattern cannot be empty"
    
    if search_type not in ("content", "filename"):
        return f"Error: search_type must be 'content' or 'filename', got '{search_type}'"
    
    # Set default path
    search_path = path or _ALLOWED_ROOT or os.getcwd()
    search_path = os.path.abspath(search_path)
    
    # Validate path is within allowed root
    if _ALLOWED_ROOT is not None:
        if not search_path.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    # Validate and cap max_results
    effective_max = 50
    if max_results is not None:
        if max_results < 1:
            return "Error: max_results must be at least 1"
        effective_max = min(max_results, 200)
    
    try:
        if search_type == "content":
            return _search_content(pattern, search_path, file_extension, effective_max)
        else:
            return _search_filename(pattern, search_path, effective_max)
    except Exception as e:
        return f"Error during search: {e}"


def _search_content(pattern: str, path: str, extension: str | None, max_results: int) -> str:
    """Search file contents for pattern using grep."""
    # Build grep command
    cmd = ["grep", "-r", "-n", "-l", "--include=*"]
    
    if extension:
        # Validate extension format
        ext = extension if extension.startswith(".") else f".{extension}"
        cmd[-1] = f"--include=*{ext}"
    
    cmd.extend([pattern, path])
    
    # Run search
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode not in (0, 1):  # 0 = matches found, 1 = no matches
        return f"Error: grep failed with code {result.returncode}: {result.stderr}"
    
    files = result.stdout.strip().split("\n") if result.stdout.strip() else []
    files = [f for f in files if f]  # Remove empty strings
    
    if not files:
        return f"No files found containing pattern '{pattern}'"
    
    # Limit results
    total_found = len(files)
    if total_found > max_results:
        files = files[:max_results]
        truncated_msg = f" (showing first {max_results} of {total_found})"
    else:
        truncated_msg = ""
    
    # Get context for each file
    output_lines = [f"Found {total_found} file(s) containing '{pattern}'{truncated_msg}:\n"]
    
    for filepath in files:
        try:
            # Get matching lines with context
            grep_result = subprocess.run(
                ["grep", "-n", "-C", "2", pattern, filepath],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if grep_result.returncode == 0:
                output_lines.append(f"\n=== {filepath} ===")
                output_lines.append(grep_result.stdout.strip())
        except Exception:
            output_lines.append(f"\n=== {filepath} ===")
            output_lines.append("(could not read file contents)")
    
    return _truncate("\n".join(output_lines))


def _search_filename(pattern: str, path: str, max_results: int) -> str:
    """Search for files by name pattern using find."""
    # Convert glob pattern to find pattern if needed
    # find uses -name for exact match, -iname for case-insensitive
    
    cmd = ["find", path, "-type", "f", "-name", pattern]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode != 0:
        return f"Error: find failed: {result.stderr}"
    
    files = result.stdout.strip().split("\n") if result.stdout.strip() else []
    files = [f for f in files if f]
    
    if not files:
        # Try case-insensitive search
        cmd = ["find", path, "-type", "f", "-iname", pattern]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        files = [f for f in files if f]
        
        if not files:
            return f"No files found matching pattern '{pattern}'"
    
    total_found = len(files)
    if total_found > max_results:
        files = files[:max_results]
        truncated_msg = f" (showing first {max_results} of {total_found})"
    else:
        truncated_msg = ""
    
    output = f"Found {total_found} file(s) matching '{pattern}'{truncated_msg}:\n"
    output += "\n".join(files)
    
    return _truncate(output)
