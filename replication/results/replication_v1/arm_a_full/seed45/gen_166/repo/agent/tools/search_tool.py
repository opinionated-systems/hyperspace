"""
Search tool: search for patterns in files using grep/ripgrep.

Provides file search capabilities to help navigate and understand codebases.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Searches file contents for regex patterns. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path). Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files.
    
    Args:
        pattern: The regex pattern to search for
        path: Directory or file to search in (defaults to allowed root)
        file_extension: Optional file extension filter (e.g., '.py')
        max_results: Maximum number of results to return
    
    Returns:
        Search results with file paths, line numbers, and matching lines
    """
    # Validate pattern
    if pattern is None or not str(pattern).strip():
        return "Error: pattern parameter is required and cannot be empty"
    pattern = str(pattern).strip()
    
    # Determine search path
    if path is None:
        if _ALLOWED_ROOT is None:
            return "Error: No search path specified and no allowed root set"
        search_path = _ALLOWED_ROOT
    else:
        search_path = str(path).strip()
    
    p = Path(search_path)
    
    if not p.is_absolute():
        return f"Error: {search_path} is not an absolute path."
    
    # Scope check
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    # Build grep command
    cmd_parts = ["grep", "-rn"]
    
    # Add file extension filter if provided
    if file_extension:
        # Ensure extension starts with * for glob pattern
        if not file_extension.startswith("*"):
            file_pattern = f"*{file_extension}"
        else:
            file_pattern = file_extension
        cmd_parts.extend(["--include", file_pattern])
    
    # Add pattern
    cmd_parts.append(pattern)
    
    # Add search path
    cmd_parts.append(str(p))
    
    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {search_path}"
        
        if result.returncode != 0 and result.stderr:
            return f"Error searching: {result.stderr}"
        
        # Process results
        lines = result.stdout.strip().split("\n")
        if not lines or lines == ['']:
            return f"No matches found for pattern '{pattern}' in {search_path}"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... [truncated, showing {max_results} of {len(lines)}+ matches] ..."
        else:
            truncated_msg = ""
        
        # Format output
        output_lines = [f"Search results for '{pattern}' in {search_path}:", "-" * 60]
        for line in lines:
            if line.strip():
                output_lines.append(line)
        
        if truncated_msg:
            output_lines.append(truncated_msg)
        
        output_lines.append(f"\nTotal matches: {len(lines)}")
        
        return "\n".join(output_lines)
        
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after 30 seconds. Try a more specific pattern or narrower scope."
    except FileNotFoundError:
        # grep not available, try with Python
        return _python_search(p, pattern, file_extension, max_results)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _python_search(
    path: Path,
    pattern: str,
    file_extension: str | None,
    max_results: int,
) -> str:
    """Fallback search using Python's re module."""
    import re
    
    results = []
    count = 0
    
    try:
        if path.is_file():
            files_to_search = [path]
        else:
            files_to_search = path.rglob("*")
        
        for file_path in files_to_search:
            if not file_path.is_file():
                continue
            
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            
            # Filter by extension
            if file_extension and not str(file_path).endswith(file_extension):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            results.append(f"{file_path}:{line_num}:{line.rstrip()}")
                            count += 1
                            if count >= max_results:
                                break
                
                if count >= max_results:
                    break
                    
            except (IOError, OSError):
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        output_lines = [f"Search results for '{pattern}' in {path}:", "-" * 60]
        output_lines.extend(results)
        
        if count >= max_results:
            output_lines.append(f"\n... [truncated, showing {max_results} matches] ...")
        
        output_lines.append(f"\nTotal matches: {count}")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"Error in Python search: {type(e).__name__}: {e}"
