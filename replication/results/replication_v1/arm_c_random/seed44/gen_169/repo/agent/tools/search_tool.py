"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality and file finding capabilities
to help the meta agent navigate and understand the codebase.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content within the codebase. "
            "Commands: find_files (by pattern), grep (search content), "
            "find_functions (find Python function definitions)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_functions"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (glob for find_files, regex for grep).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
                },
            },
            "required": ["command", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _validate_path(path: str) -> tuple[bool, str]:
    """Validate that path is absolute and within allowed root."""
    if not path:
        return False, "Path is required"
    
    p = Path(path)
    if not p.is_absolute():
        return False, f"Error: {path} is not an absolute path"
    
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    return True, ""


def _truncate_results(results: list[str], max_results: int = 50) -> str:
    """Truncate results list to max_results."""
    if len(results) > max_results:
        truncated = results[:max_results]
        return "\n".join(truncated) + f"\n... ({len(results) - max_results} more results truncated)"
    return "\n".join(results)


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    # Validate inputs
    is_valid, error_msg = _validate_path(path)
    if not is_valid:
        return error_msg
    
    if not os.path.exists(path):
        return f"Error: Path {path} does not exist"
    
    max_results = min(max(1, max_results), 100)  # Clamp between 1 and 100
    
    try:
        if command == "find_files":
            return _find_files(path, pattern, file_extension, max_results)
        elif command == "grep":
            return _grep(path, pattern, file_extension, max_results)
        elif command == "find_functions":
            return _find_functions(path, pattern, max_results)
        else:
            return f"Error: unknown command '{command}'. Valid commands: find_files, grep, find_functions"
    except Exception as e:
        return f"Error executing search: {type(e).__name__}: {e}"


def _find_files(base_path: str, pattern: str | None, file_extension: str | None, max_results: int) -> str:
    """Find files matching pattern and/or extension."""
    results = []
    
    if pattern and "*" in pattern:
        # Use glob pattern
        import glob
        search_pattern = os.path.join(base_path, "**", pattern)
        matches = glob.glob(search_pattern, recursive=True)
        results = [m for m in matches if os.path.isfile(m)]
    else:
        # Walk directory tree
        for root, dirs, files in os.walk(base_path):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            
            for filename in files:
                if filename.startswith("."):
                    continue
                
                # Check extension filter
                if file_extension and not filename.endswith(file_extension):
                    continue
                
                # Check name pattern
                if pattern and pattern not in filename:
                    continue
                
                full_path = os.path.join(root, filename)
                results.append(full_path)
                
                if len(results) >= max_results * 2:  # Collect extra for truncation message
                    break
            
            if len(results) >= max_results * 2:
                break
    
    if not results:
        return f"No files found in {base_path}" + (f" matching '{pattern}'" if pattern else "")
    
    # Make paths relative to base_path for cleaner output
    relative_results = []
    for r in results[:max_results]:
        try:
            rel = os.path.relpath(r, base_path)
            relative_results.append(rel)
        except ValueError:
            relative_results.append(r)
    
    total_found = len(results)
    output = _truncate_results(relative_results, max_results)
    return f"Found {total_found} file(s) in {base_path}:\n{output}"


def _grep(base_path: str, pattern: str | None, file_extension: str | None, max_results: int) -> str:
    """Search for pattern in file contents."""
    if not pattern:
        return "Error: 'pattern' is required for grep command"
    
    results = []
    
    # Use ripgrep if available, otherwise fall back to Python implementation
    try:
        cmd = ["rg", "-n", "--no-heading", "--color=never", pattern]
        if file_extension:
            cmd.extend(["-g", f"*{file_extension}"])
        cmd.append(base_path)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode in [0, 1]:  # 0 = matches found, 1 = no matches
            lines = result.stdout.strip().split("\n") if result.stdout else []
            lines = [l for l in lines if l]
            
            if lines:
                return f"Found {len(lines)} match(es) for '{pattern}':\n" + _truncate_results(lines, max_results)
            return f"No matches found for '{pattern}'"
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fall back to Python implementation
        pass
    
    # Python fallback
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for filename in files:
            if filename.startswith("."):
                continue
            if file_extension and not filename.endswith(file_extension):
                continue
            
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if pattern in line:
                            rel_path = os.path.relpath(filepath, base_path)
                            results.append(f"{rel_path}:{line_num}:{line.rstrip()}")
                            if len(results) >= max_results * 2:
                                break
                        if len(results) >= max_results * 2:
                            break
            except (IOError, OSError):
                continue
            
            if len(results) >= max_results * 2:
                break
        
        if len(results) >= max_results * 2:
            break
    
    if not results:
        return f"No matches found for '{pattern}'"
    
    total_found = len(results)
    output = _truncate_results(results[:max_results], max_results)
    return f"Found {total_found} match(es) for '{pattern}':\n{output}"


def _find_functions(base_path: str, function_name: str | None, max_results: int) -> str:
    """Find Python function/class definitions."""
    import re
    
    pattern = function_name if function_name else r"[a-zA-Z_][a-zA-Z0-9_]*"
    func_pattern = re.compile(rf"^\s*(?:def|class)\s+({pattern})\s*[\(:]")
    
    results = []
    
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for filename in files:
            if not filename.endswith(".py"):
                continue
            
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        match = func_pattern.match(line)
                        if match:
                            rel_path = os.path.relpath(filepath, base_path)
                            func_type = "class" if "class" in line else "def"
                            func_name = match.group(1)
                            results.append(f"{rel_path}:{line_num}:{func_type} {func_name}")
                            if len(results) >= max_results * 2:
                                break
                        if len(results) >= max_results * 2:
                            break
            except (IOError, OSError):
                continue
            
            if len(results) >= max_results * 2:
                break
        
        if len(results) >= max_results * 2:
            break
    
    if not results:
        search_desc = f" matching '{function_name}'" if function_name else ""
        return f"No Python functions/classes found{search_desc}"
    
    total_found = len(results)
    output = _truncate_results(results[:max_results], max_results)
    return f"Found {total_found} function(s)/class(es){search_desc if function_name else ''}:\n{output}"
