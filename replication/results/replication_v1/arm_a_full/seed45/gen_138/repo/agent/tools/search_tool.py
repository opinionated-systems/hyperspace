"""
Search tool: find patterns in files using grep/ripgrep.

Provides fast file searching capabilities to locate code patterns,
function definitions, and references across the codebase.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Fast way to find code patterns, function definitions, "
            "and references across the codebase. "
            "Returns matching lines with file paths and line numbers. "
            "Supports both single pattern and multi-pattern searches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple search patterns to search for simultaneously (alternative to single pattern).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path). Default: allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Filter by file extension (e.g., '.py'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["any", "function_def", "class_def"],
                    "description": "Type of search: 'any' for general pattern, 'function_def' for function definitions, 'class_def' for class definitions. Default: 'any'.",
                },
            },
        },
    }


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def _build_search_pattern(pattern: str, search_type: str) -> str:
    """Build the search pattern based on search type.
    
    Args:
        pattern: The base search pattern
        search_type: Type of search ('any', 'function_def', 'class_def')
    
    Returns:
        Modified pattern appropriate for the search type
    """
    if search_type == "function_def":
        # Match function definitions: def pattern( or async def pattern(
        return rf"^\s*(async\s+)?def\s+{re.escape(pattern)}\s*\("
    elif search_type == "class_def":
        # Match class definitions: class pattern( or class pattern:
        return rf"^\s*class\s+{re.escape(pattern)}\s*[(:]"
    else:
        # General pattern search
        return pattern


def _search_single_pattern(
    pattern: str,
    search_path: str,
    file_extension: str | None,
    case_sensitive: bool,
    max_results: int,
    search_type: str,
) -> tuple[list[str], str]:
    """Search for a single pattern and return (lines, description).
    
    Returns:
        Tuple of (list of matching lines, search description string)
    """
    # Build the actual search pattern based on type
    actual_pattern = _build_search_pattern(pattern, search_type)

    # Build command - prefer ripgrep, fallback to grep
    cmd = []
    
    # Check for ripgrep
    rg_available = subprocess.run(
        ["which", "rg"], capture_output=True
    ).returncode == 0
    
    if rg_available:
        cmd = ["rg", "--line-number", "--no-heading"]
        if not case_sensitive:
            cmd.append("-i")
        if file_extension:
            cmd.extend(["-g", f"*{file_extension}"])
        cmd.extend(["-m", str(max_results), actual_pattern, search_path])
    else:
        # Fallback to grep
        cmd = ["grep", "-r", "-n"]
        if not case_sensitive:
            cmd.append("-i")
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        cmd.extend(["-m", str(max_results), actual_pattern, search_path])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode not in (0, 1):  # 0 = matches found, 1 = no matches
        raise RuntimeError(f"Search failed with code {result.returncode}: {result.stderr}")

    output = result.stdout.strip()
    if not output:
        return [], pattern

    lines = output.split("\n")
    search_desc = f"{search_type} '{pattern}'" if search_type != "any" else f"'{pattern}'"
    return lines, search_desc


def tool_function(
    pattern: str | None = None,
    patterns: list[str] | None = None,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
    search_type: str = "any",
) -> str:
    """Search for pattern(s) in files.
    
    Supports both single pattern (pattern param) and multi-pattern (patterns param) searches.
    Uses ripgrep if available, falls back to grep.
    
    Args:
        pattern: Single search pattern (alternative to patterns)
        patterns: List of search patterns for multi-pattern search (alternative to pattern)
        path: Directory to search in
        file_extension: Filter by file extension
        case_sensitive: Whether search is case sensitive
        max_results: Maximum number of results per pattern
        search_type: Type of search ('any', 'function_def', 'class_def')
    """
    try:
        # Validate pattern input
        if pattern is None and patterns is None:
            return "Error: either 'pattern' or 'patterns' must be provided."
        if pattern is not None and patterns is not None:
            return "Error: provide either 'pattern' or 'patterns', not both."
        
        # Normalize to list of patterns
        search_patterns = patterns if patterns is not None else [pattern]
        
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: no path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            # Scope check
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        if not os.path.isdir(search_path):
            return f"Error: {search_path} is not a directory."

        # Search for each pattern
        all_results = []
        total_matches = 0
        
        for pat in search_patterns:
            try:
                lines, search_desc = _search_single_pattern(
                    pat, search_path, file_extension, case_sensitive, max_results, search_type
                )
                if lines:
                    all_results.append((search_desc, lines))
                    total_matches += len(lines)
            except RuntimeError as e:
                return f"Error: {e}"

        if not all_results:
            if len(search_patterns) == 1:
                search_desc = f"{search_type} '{search_patterns[0]}'" if search_type != "any" else f"'{search_patterns[0]}'"
                return f"No matches found for {search_desc} in {search_path}"
            else:
                return f"No matches found for any of {len(search_patterns)} patterns in {search_path}"

        # Format results
        output_parts = []
        for search_desc, lines in all_results:
            if len(lines) > max_results:
                lines = lines[:max_results]
                output_parts.append(f"Found {len(lines)}+ matches for {search_desc}:")
            else:
                output_parts.append(f"Found {len(lines)} matches for {search_desc}:")
            output_parts.append("\n".join(lines))
            output_parts.append("")  # Empty line between patterns
        
        # Remove trailing empty line
        if output_parts and output_parts[-1] == "":
            output_parts.pop()
        
        full_output = "\n".join(output_parts)
        
        if len(search_patterns) > 1:
            header = f"Multi-pattern search: {total_matches} total matches across {len(all_results)} patterns\n"
            full_output = header + full_output
        
        return _truncate(full_output)

    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"
