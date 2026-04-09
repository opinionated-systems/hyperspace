"""
Search tool: find files and search for patterns within the codebase.

Provides grep-like functionality and file finding capabilities
to help the agent locate code patterns and files.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search tool for finding files and searching content within the codebase. "
            "Commands: find_files (glob patterns), grep (search content), find_symbol (find classes/functions)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_symbol"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (glob for find_files, regex for grep/symbol).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
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


def _truncate_output(output: str, max_lines: int = 100) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {path} does not exist."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if command == "find_files":
            return _find_files(p, pattern or "*", file_extension, max_results)
        elif command == "grep":
            if pattern is None:
                return "Error: pattern required for grep."
            return _grep(p, pattern, file_extension, max_results)
        elif command == "find_symbol":
            if pattern is None:
                return "Error: pattern required for find_symbol."
            return _find_symbol(p, pattern, file_extension, max_results)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(base: Path, pattern: str, extension: str | None, max_results: int) -> str:
    """Find files matching a glob pattern."""
    if extension:
        glob_pattern = f"**/*{extension}"
    else:
        glob_pattern = f"**/{pattern}"

    matches = list(base.glob(glob_pattern))
    files = [str(f.relative_to(base)) for f in matches if f.is_file()][:max_results]

    if not files:
        return f"No files found matching '{pattern}' in {base}"

    result = f"Found {len(files)} file(s) in {base}:\n"
    result += "\n".join(files)
    return result


def _grep(base: Path, pattern: str, extension: str | None, max_results: int) -> str:
    """Search for pattern in file contents using grep."""
    # Convert extension to glob pattern if provided
    include_pattern = f"*{extension}" if extension else "*"
    # Use -F for fixed strings (literal match) instead of regex
    cmd = ["grep", "-r", "-n", "-I", "-F", "--include", include_pattern, pattern, str(base)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        lines = [l for l in lines if l][:max_results]

        if not lines:
            return f"No matches found for '{pattern}' in {base}"

        output = f"Found {len(lines)} match(es) for '{pattern}':\n"
        output += _truncate_output("\n".join(lines))
        return output
    except subprocess.TimeoutExpired:
        return "Error: search timed out (30s limit)"
    except Exception as e:
        return f"Error running grep: {e}"


def _find_symbol(base: Path, symbol: str, extension: str | None, max_results: int) -> str:
    """Find class or function definitions matching the symbol."""
    ext = extension or ".py"
    include_pattern = f"*{ext}"
    # Search for class or function definitions using regex
    # Note: grep -E uses extended regex, \\b is word boundary
    patterns = [
        f"^class\\s+{symbol}([^a-zA-Z_]|$)",  # class definition
        f"^def\\s+{symbol}([^a-zA-Z_]|$)",     # function definition
        f"^{symbol}\\s*[=:]",                   # variable assignment
    ]

    all_matches = []
    for pat in patterns:
        cmd = ["grep", "-r", "-n", "-I", "-E", "--include", include_pattern, pat, str(base)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.stdout:
                all_matches.extend(result.stdout.strip().split("\n"))
        except Exception:
            pass

    matches = [m for m in all_matches if m][:max_results]

    if not matches:
        return f"No symbol '{symbol}' found in {base}"

    output = f"Found {len(matches)} definition(s) for '{symbol}':\n"
    output += _truncate_output("\n".join(matches))
    return output
