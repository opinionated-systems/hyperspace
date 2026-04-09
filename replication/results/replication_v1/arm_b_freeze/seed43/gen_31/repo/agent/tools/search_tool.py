"""
Search tool: search for patterns in files using grep/ripgrep.

Provides file search capabilities to help navigate large codebases.
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
            "Searches file contents for matching patterns. "
            "Returns matching file paths with line numbers and context."
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
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
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


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a file search."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {p} does not exist."

        # Build grep command
        cmd = ["grep", "-r", "-n", "-i"]  # recursive, line numbers, case insensitive
        
        # Add context lines
        cmd.extend(["-B", "2", "-A", "2"])  # 2 lines before and after
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(p))
        
        # Add file extension filter if specified
        if file_extension:
            # Use find with grep for extension filtering
            find_cmd = [
                "find", str(p), "-type", "f", "-name", f"*{file_extension}",
                "-exec", "grep", "-H", "-n", "-i", "-B", "2", "-A", "2", pattern, "{}", "+",
                "2>/dev/null"
            ]
            result = subprocess.run(
                " ".join(find_cmd),
                capture_output=True,
                text=True,
                shell=True,
                timeout=30,
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more results)")
            return _truncate("\n".join(lines))
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}' in {p}"
        else:
            return f"Error searching: {result.stderr}"

    except subprocess.TimeoutExpired:
        return f"Error: search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
