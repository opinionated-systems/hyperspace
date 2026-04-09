"""
Search tool: search for patterns in files using grep.

Provides file content search capabilities to help agents explore codebases.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep or find files by name. "
            "Supports searching within specific directories or files. "
            "Returns matching lines with file paths and line numbers. "
            "Use search_type='content' for grep (default) or search_type='filename' for find."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported for content, glob pattern for filename).",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to directory or file to search in. Defaults to current directory.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive. Default is True.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Type of search: 'content' searches file contents with grep, 'filename' searches for files by name pattern.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = Path(root).resolve()


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = True,
    search_type: str = "content",
) -> str:
    """Execute a search using grep (content) or find (filename)."""
    try:
        # Default to current directory if no path provided
        if path is None:
            path = "."
        
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = p.resolve()
            if not str(resolved).startswith(str(_ALLOWED_ROOT)):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        # Handle filename search using find
        if search_type == "filename":
            return _search_filename(pattern, p, file_extension, case_sensitive)
        
        # Default: content search using grep
        return _search_content(pattern, p, file_extension, case_sensitive)
            
    except subprocess.TimeoutExpired:
        return "Error: search timed out (max 30 seconds)"
    except Exception as e:
        return f"Error: {e}"


def _search_content(
    pattern: str,
    p: Path,
    file_extension: str | None,
    case_sensitive: bool,
) -> str:
    """Search file contents using grep."""
    # Build grep command
    cmd = ["grep", "-r", "-n"]
    
    if not case_sensitive:
        cmd.append("-i")
    
    # Add pattern
    cmd.append(pattern)
    
    # Add path
    cmd.append(str(p))
    
    # Add file extension filter if provided
    if file_extension:
        cmd.extend(["--include", f"*{file_extension}"])
    
    # Execute search
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode == 0:
        # Found matches
        lines = result.stdout.strip().split("\n")
        if len(lines) > 100:
            # Truncate if too many results
            truncated = lines[:50] + [f"... ({len(lines) - 100} more matches) ..."] + lines[-50:]
            return "Search results (truncated):\n" + "\n".join(truncated)
        return f"Search results ({len(lines)} matches):\n" + result.stdout
    elif result.returncode == 1:
        # No matches found
        return f"No matches found for pattern '{pattern}' in {path}"
    else:
        # Error occurred
        return f"Search error: {result.stderr}"


def _search_filename(
    pattern: str,
    p: Path,
    file_extension: str | None,
    case_sensitive: bool,
) -> str:
    """Search for files by name pattern using find."""
    # Build find command
    cmd = ["find", str(p), "-type", "f"]
    
    # Add name pattern matching
    if case_sensitive:
        cmd.extend(["-name", pattern])
    else:
        cmd.extend(["-iname", pattern])
    
    # Add file extension filter if provided (additional filter)
    if file_extension:
        # Combine with -a (and) operator for both conditions
        cmd = ["find", str(p), "-type", "f", "(", "-name", f"*{file_extension}", ")"]
        if case_sensitive:
            cmd.extend(["-a", "-name", pattern])
        else:
            cmd.extend(["-a", "-iname", pattern])
    
    # Execute search
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode == 0:
        # Success (find returns 0 even with no matches)
        lines = [line for line in result.stdout.strip().split("\n") if line]
        if not lines:
            return f"No files found matching pattern '{pattern}' in {p}"
        if len(lines) > 100:
            # Truncate if too many results
            truncated = lines[:50] + [f"... ({len(lines) - 100} more files) ..."] + lines[-50:]
            return f"Found files ({len(lines)} total, truncated):\n" + "\n".join(truncated)
        return f"Found {len(lines)} files:\n" + "\n".join(lines)
    else:
        # Error occurred
        return f"Search error: {result.stderr}"
