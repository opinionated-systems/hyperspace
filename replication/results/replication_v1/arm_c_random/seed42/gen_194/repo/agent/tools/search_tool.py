"""
Search tool: search for patterns in files using grep-like functionality.

Provides file content search capabilities to complement the editor tool.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files. Supports text search, regex patterns, "
            "and file filtering. Returns matching lines with context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (text or regex).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Absolute path.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter by (e.g., '*.py'). Optional.",
                },
                "use_regex": {
                    "type": "boolean",
                    "description": "Whether to treat pattern as regex. Default: false.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show. Default: 2.",
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


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    use_regex: bool = False,
    context_lines: int = 2,
) -> str:
    """Execute a search for the given pattern."""
    logger.info(f"Search: pattern={pattern}, path={path}")
    
    try:
        p = Path(path)

        if not p.is_absolute():
            logger.warning(f"Relative path rejected: {path}")
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                logger.warning(f"Access denied: {resolved} outside {_ALLOWED_ROOT}")
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        results = []
        
        if p.is_file():
            # Search in single file
            results = _search_file(p, pattern, use_regex, context_lines)
        else:
            # Search in directory
            results = _search_directory(p, pattern, file_pattern, use_regex, context_lines)
        
        if not results:
            return f"No matches found for '{pattern}' in {path}"
        
        # Format results
        output = [f"Found {len(results)} match(es) for '{pattern}':\n"]
        for result in results[:50]:  # Limit to 50 results
            output.append(result)
        
        if len(results) > 50:
            output.append(f"\n... and {len(results) - 50} more matches")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error: {e}"


def _search_file(
    file_path: Path,
    pattern: str,
    use_regex: bool,
    context_lines: int,
) -> list[str]:
    """Search for pattern in a single file."""
    results = []
    
    try:
        content = file_path.read_text()
        lines = content.split("\n")
        
        if use_regex:
            regex = re.compile(pattern)
            matcher = lambda line: regex.search(line)
        else:
            matcher = lambda line: pattern in line
        
        for i, line in enumerate(lines):
            if matcher(line):
                # Get context
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                
                context = []
                for j in range(start, end):
                    marker = ">>> " if j == i else "    "
                    context.append(f"{marker}{j + 1:4}: {lines[j]}")
                
                results.append(f"\n{file_path}:{i + 1}")
                results.append("\n".join(context))
    
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
    
    return results


def _search_directory(
    dir_path: Path,
    pattern: str,
    file_pattern: str | None,
    use_regex: bool,
    context_lines: int,
) -> list[str]:
    """Search for pattern in all files under directory."""
    results = []
    
    # Use find to get files
    if file_pattern:
        cmd = ["find", str(dir_path), "-type", "f", "-name", file_pattern, "-not", "-path", "*/\.*"]
    else:
        cmd = ["find", str(dir_path), "-type", "f", "-not", "-path", "*/\.*"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        
        for file_str in files:
            if not file_str:
                continue
            file_results = _search_file(Path(file_str), pattern, use_regex, context_lines)
            results.extend(file_results)
    
    except subprocess.TimeoutExpired:
        logger.warning("Search timeout")
    except Exception as e:
        logger.warning(f"Search error: {e}")
    
    return results
