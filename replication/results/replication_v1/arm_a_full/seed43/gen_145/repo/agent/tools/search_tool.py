"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns within files.
Supports regex patterns and can search recursively through directories.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files. "
            "Supports regex patterns and recursive directory search. "
            "Returns matching lines with file paths and line numbers. "
            "Can filter by file patterns (e.g., '*.py') and show context lines."
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
                    "description": "Absolute path to file or directory to search in.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, search recursively in directories. Default: true.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "If false, perform case-insensitive search. Default: true.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py', '*.js'). Default: search all files.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before and after each match. Default: 0.",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_under_root(path: Path) -> bool:
    """Check if path is under the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    try:
        Path(_ALLOWED_ROOT).resolve() in Path(path).resolve().parents or \
        Path(_ALLOWED_ROOT).resolve() == Path(path).resolve()
    except (OSError, ValueError):
        return False
    return True


def _should_skip_file(p: Path) -> bool:
    """Check if file should be skipped (binary, hidden, etc.)."""
    name = p.name
    # Skip hidden files and common non-source directories
    if name.startswith('.') or name in {'__pycache__', 'node_modules', '.git', '.venv', 'venv'}:
        return True
    # Skip common binary extensions
    binary_exts = {'.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.dat', '.db', '.sqlite', '.jpg', '.png', '.gif', '.pdf'}
    if p.suffix.lower() in binary_exts:
        return True
    return False


def _search_file(p: Path, pattern: re.Pattern, max_results: int, context_lines: int = 0) -> list[tuple[str, int, str, list[str], list[str]]]:
    """Search a single file for the pattern.
    
    Returns list of (path, line_num, line_content, before_context, after_context) tuples.
    """
    results = []
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                # Get context lines
                before = []
                after = []
                if context_lines > 0:
                    start_idx = max(0, i - context_lines - 1)
                    before = [l.rstrip('\n') for l in lines[start_idx:i-1]]
                    end_idx = min(len(lines), i + context_lines)
                    after = [l.rstrip('\n') for l in lines[i:end_idx]]
                
                results.append((str(p), i, line.rstrip('\n'), before, after))
                if len(results) >= max_results:
                    break
    except (IOError, OSError, UnicodeDecodeError):
        pass
    return results


def _matches_file_pattern(filename: str, pattern: str | None) -> bool:
    """Check if filename matches the glob pattern."""
    if pattern is None:
        return True
    import fnmatch
    return fnmatch.fnmatch(filename, pattern)


def tool_function(
    pattern: str,
    path: str,
    recursive: bool = True,
    case_sensitive: bool = True,
    max_results: int = 50,
    file_pattern: str | None = None,
    context_lines: int = 0,
) -> str:
    """Execute a search for the pattern in the specified path."""
    # Validate inputs
    if not pattern:
        return "Error: pattern is required"
    if not path:
        return "Error: path is required"
    
    p = Path(path)
    
    if not p.is_absolute():
        return f"Error: {path} is not an absolute path. Please provide an absolute path."
    
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if _ALLOWED_ROOT and not _is_under_root(p):
        return f"Error: {p} is outside the allowed root directory."
    
    # Compile regex pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    results = []
    
    if p.is_file():
        if _matches_file_pattern(p.name, file_pattern):
            results = _search_file(p, regex, max_results, context_lines)
    elif p.is_dir():
        if recursive:
            for root, dirs, files in os.walk(p):
                # Filter out hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules', '.git'}]
                for filename in files:
                    if not _matches_file_pattern(filename, file_pattern):
                        continue
                    file_path = Path(root) / filename
                    if _should_skip_file(file_path):
                        continue
                    file_results = _search_file(file_path, regex, max_results - len(results), context_lines)
                    results.extend(file_results)
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break
        else:
            # Non-recursive: only search files in the directory
            for item in p.iterdir():
                if item.is_file() and not _should_skip_file(item):
                    if not _matches_file_pattern(item.name, file_pattern):
                        continue
                    file_results = _search_file(item, regex, max_results - len(results), context_lines)
                    results.extend(file_results)
                    if len(results) >= max_results:
                        break
    
    # Format results
    if not results:
        return f"No matches found for pattern '{pattern}' in {p}"
    
    lines = [f"Found {len(results)} match(es) for pattern '{pattern}':\n"]
    
    current_file = None
    for file_path, line_num, content, before, after in results[:max_results]:
        # Add file header if new file
        if file_path != current_file:
            if current_file is not None:
                lines.append("")  # Blank line between files
            current_file = file_path
        
        # Show context lines before
        for i, ctx_line in enumerate(before, start=line_num - len(before)):
            ctx_content = ctx_line
            if len(ctx_content) > 200:
                ctx_content = ctx_content[:100] + " ... " + ctx_content[-100:]
            lines.append(f"{file_path}:{i}: {ctx_content}")
        
        # Show the match line
        if len(content) > 200:
            content = content[:100] + " ... " + content[-100:]
        lines.append(f"{file_path}:{line_num}: {content}")
        
        # Show context lines after
        for i, ctx_line in enumerate(after, start=line_num + 1):
            ctx_content = ctx_line
            if len(ctx_content) > 200:
                ctx_content = ctx_content[:100] + " ... " + ctx_content[-100:]
            lines.append(f"{file_path}:{i}: {ctx_content}")
    
    if len(results) > max_results:
        lines.append(f"\n... and {len(results) - max_results} more matches (limit reached)")
    
    return "\n".join(lines)
