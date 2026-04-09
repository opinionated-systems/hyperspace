"""
Search tool: find files and search content within the codebase.

Provides grep-like functionality to search for patterns in files.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files. "
            "Commands: grep (search content), find (search filenames), count (count lines). "
            "Supports regex patterns and file filtering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find", "count"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for grep, glob for find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern filter (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                    "default": 50,
                },
            },
            "required": ["command", "pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate_output(output: str, max_len: int = 10000) -> str:
    """Truncate output if too long."""
    if len(output) > max_len:
        lines = output.split("\n")
        # Keep first half and last half of lines
        half = len(lines) // 2
        return "\n".join(lines[:half]) + f"\n... ({len(lines) - half * 2} lines omitted) ...\n" + "\n".join(lines[-half:])
    return output


def tool_function(
    command: str,
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        if command == "grep":
            return _grep(p, pattern, file_pattern, max_results)
        elif command == "find":
            return _find(p, pattern, max_results)
        elif command == "count":
            return _count(p, pattern)
        else:
            return f"Error: unknown command {command}"

    except Exception as e:
        return f"Error: {e}"


def _grep(path: Path, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Search for pattern in file contents."""
    cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", pattern, str(path)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        lines = [l for l in lines if l]  # Remove empty lines
        
        if not lines:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        total = len(lines)
        if total > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total - max_results} more results truncated) ..."
        else:
            truncated_msg = ""
        
        output = "\n".join(lines) + truncated_msg
        return f"Found {total} match(es) for '{pattern}':\n{output}"
        
    except subprocess.TimeoutExpired:
        return f"Error: grep timed out after 30s"
    except Exception as e:
        return f"Error running grep: {e}"


def _find(path: Path, pattern: str, max_results: int) -> str:
    """Find files by name pattern."""
    cmd = ["find", str(path), "-name", pattern, "-type", "f"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        lines = [l for l in lines if l]
        
        if not lines:
            return f"No files found matching '{pattern}' in {path}"
        
        total = len(lines)
        if total > max_results:
            lines = lines[:max_results]
            truncated_msg = f"\n... ({total - max_results} more results truncated) ..."
        else:
            truncated_msg = ""
        
        output = "\n".join(lines) + truncated_msg
        return f"Found {total} file(s) matching '{pattern}':\n{output}"
        
    except subprocess.TimeoutExpired:
        return f"Error: find timed out after 30s"
    except Exception as e:
        return f"Error running find: {e}"


def _count(path: Path, pattern: str) -> str:
    """Count occurrences of a pattern in file contents."""
    cmd = ["grep", "-r", "-c", "-I", pattern, str(path)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        lines = [l for l in lines if l and ":" in l]
        
        if not lines:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        # Parse counts
        total_count = 0
        file_counts = []
        for line in lines:
            parts = line.rsplit(":", 1)
            if len(parts) == 2:
                filepath, count_str = parts
                try:
                    count = int(count_str)
                    total_count += count
                    if count > 0:
                        file_counts.append((filepath, count))
                except ValueError:
                    continue
        
        # Sort by count descending
        file_counts.sort(key=lambda x: x[1], reverse=True)
        
        output_lines = [f"Total occurrences: {total_count}", ""]
        output_lines.append("Breakdown by file:")
        for filepath, count in file_counts[:20]:  # Show top 20
            output_lines.append(f"  {filepath}: {count}")
        
        if len(file_counts) > 20:
            output_lines.append(f"  ... and {len(file_counts) - 20} more files")
        
        return "\n".join(output_lines)
        
    except subprocess.TimeoutExpired:
        return f"Error: count timed out after 30s"
    except Exception as e:
        return f"Error running count: {e}"
