"""
Search tool: find patterns in files using grep/ripgrep.

Provides fast file searching capabilities to locate code patterns,
function definitions, and references across the codebase.
"""

from __future__ import annotations

import os
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
            "Use search_type='definitions' to find function/class definitions."
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
                    "enum": ["pattern", "definitions"],
                    "description": "Type of search: 'pattern' for general search, 'definitions' for function/class definitions.",
                },
            },
            "required": ["pattern"],
        },
    }


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
    search_type: str = "pattern",
) -> str:
    """Search for pattern in files.
    
    Uses ripgrep if available, falls back to grep.
    When search_type='definitions', searches for function/class definitions
    matching the pattern (e.g., 'def pattern' or 'class pattern').
    """
    try:
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

        # Build command - prefer ripgrep, fallback to grep
        cmd = []
        
        # Check for ripgrep
        rg_available = subprocess.run(
            ["which", "rg"], capture_output=True
        ).returncode == 0
        
        # Adjust pattern for definition search
        search_pattern = pattern
        if search_type == "definitions":
            # For Python files, search for function/class definitions
            if file_extension == ".py" or file_extension is None:
                # Match 'def pattern(' or 'class pattern(' or 'async def pattern('
                search_pattern = f"^(async\s+)?(def|class)\s+{pattern}\\s*\\("
            else:
                # Generic definition search for other languages
                search_pattern = f"(def|class|function|fn)\s+{pattern}"
        
        if rg_available:
            cmd = ["rg", "--line-number", "--no-heading"]
            if not case_sensitive:
                cmd.append("-i")
            if file_extension:
                cmd.extend(["-g", f"*{file_extension}"])
            # Use regex mode for definition search
            if search_type == "definitions":
                cmd.append("--regexp")
            cmd.extend(["-m", str(max_results), search_pattern, search_path])
        else:
            # Fallback to grep
            cmd = ["grep", "-r", "-n"]
            if not case_sensitive:
                cmd.append("-i")
            if file_extension:
                cmd.extend(["--include", f"*{file_extension}"])
            # Use extended regex for definition search
            if search_type == "definitions":
                cmd.append("-E")
            cmd.extend(["-m", str(max_results), search_pattern, search_path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode not in (0, 1):  # 0 = matches found, 1 = no matches
            return f"Error: search failed with code {result.returncode}: {result.stderr}"

        output = result.stdout.strip()
        if not output:
            search_desc = "definitions" if search_type == "definitions" else "pattern"
            return f"No {search_desc} found for '{pattern}' in {search_path}"

        lines = output.split("\n")
        if len(lines) > max_results:
            lines = lines[:max_results]
            output = "\n".join(lines) + f"\n... ({len(lines)}+ total matches)"

        search_desc = "definitions" if search_type == "definitions" else "matches"
        return _truncate(f"Found {len(lines)} {search_desc} for '{pattern}':\n{output}")

    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"
