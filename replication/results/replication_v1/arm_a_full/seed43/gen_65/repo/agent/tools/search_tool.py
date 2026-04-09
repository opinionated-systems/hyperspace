"""
Search tool: search for patterns in files using grep.

Provides a convenient way to search file contents without
needing to construct complex bash grep commands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep, or find files by name. "
            "Supports regex patterns, file filtering, and recursive search. "
            "Returns matching lines with line numbers for content search, "
            "or file paths for file name search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to current directory.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to filter files (e.g., '*.py').",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories. Default: true.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search. Default: true.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
                "find_files": {
                    "type": "boolean",
                    "description": "If true, search for files by name instead of content. Default: false.",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = True,
    max_results: int = 50,
    find_files: bool = False,
) -> str:
    """Search for a pattern in files, or find files by name."""
    if not pattern:
        return "Error: pattern is required"

    p = Path(path)
    if not p.exists():
        return f"Error: path '{path}' does not exist"

    # If find_files is True, search for files by name instead of content
    if find_files:
        return _find_files_by_name(pattern, p, file_pattern, recursive, case_sensitive, max_results)

    # Build grep command for content search
    cmd = ["grep"]
    
    # Add options
    if recursive:
        cmd.append("-r")
    if not case_sensitive:
        cmd.append("-i")
    # Always show line numbers
    cmd.append("-n")
    # Add pattern
    cmd.append(pattern)
    
    # Add path
    cmd.append(str(p))
    
    # Add file pattern if specified
    if file_pattern:
        cmd.extend(["--include", file_pattern])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in '{path}'"
        
        if result.returncode != 0 and result.stderr:
            return f"Error: {result.stderr.strip()}"
        
        lines = result.stdout.strip().split("\n")
        
        # Filter out empty lines
        lines = [line for line in lines if line.strip()]
        
        if not lines:
            return f"No matches found for pattern '{pattern}' in '{path}'"
        
        total = len(lines)
        
        if total > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... (truncated, showing {max_results} of {total} matches)"
        else:
            truncated_msg = ""
        
        output = f"Found {total} match(es) for pattern '{pattern}':\n" + "\n".join(lines) + truncated_msg
        return output
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except FileNotFoundError:
        return "Error: grep command not found"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _find_files_by_name(
    pattern: str,
    path: Path,
    file_pattern: str | None = None,
    recursive: bool = True,
    case_sensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Find files by name pattern using find command."""
    import fnmatch
    import os
    
    matches = []
    
    # Determine the search depth
    if recursive:
        # Walk the directory tree
        for root, dirs, files in os.walk(path):
            for filename in files:
                # Check file_pattern first if specified
                if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                    continue
                
                # Check name pattern
                if case_sensitive:
                    if fnmatch.fnmatch(filename, pattern):
                        matches.append(os.path.join(root, filename))
                else:
                    if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                        matches.append(os.path.join(root, filename))
                
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break
    else:
        # Only search in the specified directory
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    filename = entry.name
                    # Check file_pattern first if specified
                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue
                    
                    # Check name pattern
                    if case_sensitive:
                        if fnmatch.fnmatch(filename, pattern):
                            matches.append(str(entry.path))
                    else:
                        if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                            matches.append(str(entry.path))
                    
                    if len(matches) >= max_results:
                        break
        except PermissionError:
            return f"Error: Permission denied accessing '{path}'"
    
    if not matches:
        return f"No files found matching name pattern '{pattern}' in '{path}'"
    
    total = len(matches)
    if total > max_results:
        matches = matches[:max_results]
        truncated_msg = f"\n... (truncated, showing {max_results} of {total} matches)"
    else:
        truncated_msg = ""
    
    output = f"Found {total} file(s) matching name pattern '{pattern}':\n" + "\n".join(matches) + truncated_msg
    return output
