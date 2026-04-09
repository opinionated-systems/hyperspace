"""
Search tool: search for patterns in files using grep/ripgrep.

Provides file search capabilities to help the meta-agent find code patterns,
function definitions, and references across the codebase.
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
            "Can search for text patterns, function definitions, or file names. "
            "Results are limited to avoid overwhelming output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex or plain text).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.py', '*.json'). Optional.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case sensitive. Default: false.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None
_MAX_RESULTS = 50  # Limit results to avoid overwhelming output
_MAX_OUTPUT_LEN = 10000


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Execute a search command."""
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT is None:
                return "Error: No search path specified and no allowed root set."
            search_path = _ALLOWED_ROOT
        else:
            search_path = os.path.abspath(path)
            # Scope check
            if _ALLOWED_ROOT is not None:
                if not search_path.startswith(_ALLOWED_ROOT):
                    return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"

        # Check if path exists
        if not os.path.exists(search_path):
            return f"Error: Path does not exist: {search_path}"

        # Build grep command
        cmd = ["grep", "-r", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add file pattern if specified
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        # Add search path
        cmd.append(search_path)

        # Run search
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            # grep not available, try with find + grep
            return _fallback_search(pattern, search_path, file_pattern, case_sensitive)
        except subprocess.TimeoutExpired:
            return "Error: Search timed out (30s limit). Try a more specific pattern."

        if result.returncode not in (0, 1):  # 0 = matches found, 1 = no matches
            return f"Error: Search failed: {result.stderr}"

        lines = result.stdout.strip().split("\n") if result.stdout else []
        
        if not lines or lines == ['']:
            return f"No matches found for '{pattern}' in {search_path}"

        # Limit results
        total_matches = len(lines)
        if total_matches > _MAX_RESULTS:
            lines = lines[:_MAX_RESULTS]
            truncated_msg = f"\n... ({total_matches - _MAX_RESULTS} more results truncated)"
        else:
            truncated_msg = ""

        output = "\n".join(lines)
        
        # Truncate if too long
        if len(output) > _MAX_OUTPUT_LEN:
            half = _MAX_OUTPUT_LEN // 2
            output = output[:half] + "\n... [output truncated] ...\n" + output[-half:]

        return f"Found {total_matches} match(es) for '{pattern}':\n{output}{truncated_msg}"

    except Exception as e:
        return f"Error: {e}"


def _fallback_search(
    pattern: str,
    search_path: str,
    file_pattern: str | None,
    case_sensitive: bool,
) -> str:
    """Fallback search using Python if grep is not available."""
    try:
        import fnmatch
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        matches = []
        
        for root, dirs, files in os.walk(search_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                    continue
                
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append(f"{filepath}:{line_num}:{line.rstrip()}")
                                if len(matches) >= _MAX_RESULTS:
                                    break
                        if len(matches) >= _MAX_RESULTS:
                            break
                except (IOError, OSError):
                    continue
            
            if len(matches) >= _MAX_RESULTS:
                break

        if not matches:
            return f"No matches found for '{pattern}' in {search_path}"

        total = len(matches)
        if total >= _MAX_RESULTS:
            truncated_msg = f"\n... (more results may exist)"
        else:
            truncated_msg = ""

        output = "\n".join(matches[:_MAX_RESULTS])
        return f"Found {total}+ match(es) for '{pattern}':\n{output}{truncated_msg}"

    except Exception as e:
        return f"Error in fallback search: {e}"
