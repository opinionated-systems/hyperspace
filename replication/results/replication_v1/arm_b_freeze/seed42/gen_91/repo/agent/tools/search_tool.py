"""
Search tool: search for patterns in files using grep/ripgrep.

Provides code search capabilities to find patterns across the codebase.
"""

from __future__ import annotations

import subprocess
import shutil


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for patterns in files using grep or ripgrep. Searches file contents for code patterns, function names, variable names, etc. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter by (e.g., '*.py', '*.js'). Default: all files",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: false)",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Search for a pattern in files.

    Args:
        pattern: The search pattern (regex supported)
        path: Directory or file to search in
        file_pattern: Optional file pattern filter (e.g., '*.py')
        case_sensitive: Whether search is case sensitive

    Returns:
        Search results with file paths and matching lines
    """
    # Prefer ripgrep if available, fall back to grep
    use_ripgrep = shutil.which("rg") is not None

    try:
        if use_ripgrep:
            cmd = ["rg", "--line-number", "--with-filename"]
            if not case_sensitive:
                cmd.append("-i")
            if file_pattern:
                cmd.extend(["-g", file_pattern])
            cmd.extend([pattern, path])
        else:
            # Use grep as fallback
            cmd = ["grep", "-r", "-n"]
            if not case_sensitive:
                cmd.append("-i")
            if file_pattern:
                # grep --include for file patterns
                cmd.extend(["--include", file_pattern])
            cmd.extend([pattern, path])

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
                return (
                    f"Found {len(lines)} matches (showing first 50 and last 50):\n\n"
                    + "\n".join(lines[:50])
                    + "\n... [truncated] ...\n"
                    + "\n".join(lines[-50:])
                )
            return f"Found {len(lines)} matches:\n\n" + result.stdout
        elif result.returncode == 1:
            # No matches found (grep returns 1 for no matches)
            return f"No matches found for pattern: {pattern}"
        else:
            # Error
            return f"Search error (exit code {result.returncode}): {result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error during search: {e}"
