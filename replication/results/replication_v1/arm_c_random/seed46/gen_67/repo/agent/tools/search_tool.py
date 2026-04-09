"""
Search tool: find files and search for patterns in the codebase.

Provides grep-like functionality and file finding capabilities.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and patterns in the codebase. "
            "Commands: find_files (by name pattern), grep (search content), "
            "and find_symbol (search for function/class definitions)."
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
                    "description": "Absolute path to search root directory.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename glob for find_files, regex for grep/symbol).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path_allowed(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True, ""
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_output(output: str, max_lines: int = 50, max_chars: int = 5000) -> str:
    """Truncate output to avoid overwhelming the LLM."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"\n... ({len(lines) - max_lines} more lines) ..."]
    result = "\n".join(lines)
    if len(result) > max_chars:
        half = max_chars // 2
        result = result[:half] + "\n... (output truncated) ...\n" + result[-half:]
    return result


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_extension: str | None = None,
) -> str:
    """Execute a search command."""
    allowed, error_msg = _check_path_allowed(path)
    if not allowed:
        return error_msg

    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist."
    if not p.is_dir():
        return f"Error: {path} is not a directory."

    try:
        if command == "find_files":
            return _find_files(p, pattern, file_extension)
        elif command == "grep":
            return _grep(p, pattern, file_extension)
        elif command == "find_symbol":
            return _find_symbol(p, pattern, file_extension)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _find_files(root: Path, pattern: str, extension: str | None) -> str:
    """Find files matching a glob pattern."""
    if extension:
        # Use find with name and extension
        cmd = [
            "find", str(root), "-type", "f",
            "-name", f"*{pattern}*",
        ]
        if extension:
            cmd.extend(["-name", f"*{extension}"])
    else:
        cmd = ["find", str(root), "-type", "f", "-name", f"*{pattern}*"]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=30
    )

    if result.returncode != 0:
        return f"Error running find: {result.stderr}"

    files = [f for f in result.stdout.strip().split("\n") if f]
    if not files:
        return f"No files found matching '{pattern}' in {root}"

    output = f"Found {len(files)} file(s) matching '{pattern}':\n" + "\n".join(files)
    return _truncate_output(output)


def _grep(root: Path, pattern: str, extension: str | None) -> str:
    """Search file contents for a regex pattern."""
    # Build grep command
    cmd = ["grep", "-r", "-n", "-I", "--include", extension or "*"] if extension else ["grep", "-r", "-n", "-I"]
    
    # Add exclude patterns for common non-source directories
    for exclude in [".git", "__pycache__", "node_modules", ".venv", "venv"]:
        cmd.extend(["--exclude-dir", exclude])
    
    cmd.extend([pattern, str(root)])

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=30
    )

    # grep returns 1 when no matches found, which is not an error
    if result.returncode not in (0, 1):
        return f"Error running grep: {result.stderr}"

    lines = [l for l in result.stdout.strip().split("\n") if l]
    if not lines:
        return f"No matches found for '{pattern}' in {root}"

    output = f"Found {len(lines)} match(es) for '{pattern}':\n" + "\n".join(lines)
    return _truncate_output(output)


def _find_symbol(root: Path, symbol: str, extension: str | None) -> str:
    """Search for function/class definitions."""
    # Common patterns for Python definitions
    patterns = [
        f"^class\\s+{symbol}\\b",
        f"^def\\s+{symbol}\\b",
        f"^{symbol}\\s*[=:]",
    ]

    all_results = []
    for pat in patterns:
        cmd = ["grep", "-r", "-n", "-I"]
        if extension:
            cmd.extend(["--include", extension])
        for exclude in [".git", "__pycache__", "node_modules", ".venv", "venv"]:
            cmd.extend(["--exclude-dir", exclude])
        cmd.extend([pat, str(root)])

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30
        )
        if result.returncode == 0:
            all_results.extend([l for l in result.stdout.strip().split("\n") if l])

    if not all_results:
        return f"No symbol definitions found for '{symbol}' in {root}"

    output = f"Found {len(all_results)} definition(s) for '{symbol}':\n" + "\n".join(all_results)
    return _truncate_output(output)
