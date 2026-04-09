"""
Search tool: search for files and content within the repository.

Provides grep-like functionality to find files and search content.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "search",
        "description": "Search for files or content within the repository. Supports finding files by name pattern or searching file contents with grep-like functionality.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search in",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename glob or content regex)",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["filename", "content"],
                    "description": "Type of search: 'filename' to find files by name pattern, 'content' to search within file contents",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt')",
                },
            },
            "required": ["path", "pattern", "search_type"],
        },
    }


def tool_function(
    path: str,
    pattern: str,
    search_type: str,
    file_extension: str | None = None,
) -> str:
    """Search for files or content.

    Args:
        path: Directory path to search in
        pattern: Search pattern
        search_type: 'filename' or 'content'
        file_extension: Optional file extension filter

    Returns:
        Search results as formatted string
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a directory"

    try:
        if search_type == "filename":
            return _search_by_filename(path, pattern, file_extension)
        elif search_type == "content":
            return _search_by_content(path, pattern, file_extension)
        else:
            return f"Error: Invalid search_type '{search_type}'. Use 'filename' or 'content'."
    except Exception as e:
        return f"Error during search: {e}"


def _search_by_filename(path: str, pattern: str, file_extension: str | None) -> str:
    """Find files by name pattern."""
    results = []
    path_obj = Path(path)

    # Build glob pattern
    if file_extension:
        glob_pattern = f"**/*{pattern}*{file_extension}"
    else:
        glob_pattern = f"**/*{pattern}*"

    try:
        for file_path in path_obj.glob(glob_pattern):
            if file_path.is_file():
                results.append(str(file_path.relative_to(path_obj)))
    except Exception as e:
        return f"Error during filename search: {e}"

    if not results:
        return f"No files found matching pattern '{pattern}'"

    # Limit results
    if len(results) > 50:
        results = results[:50]
        results.append(f"... ({len(results)} total matches, showing first 50)")

    return "Found files:\n" + "\n".join(results)


def _search_by_content(path: str, pattern: str, file_extension: str | None) -> str:
    """Search file contents using grep."""
    cmd = ["grep", "-r", "-n", "-l", pattern, path]

    if file_extension:
        cmd.extend(["--include", f"*{file_extension}"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0 and result.returncode != 1:
            # grep returns 1 when no matches found
            return f"No content matches found for pattern '{pattern}'"

        files = result.stdout.strip().split("\n") if result.stdout.strip() else []

        if not files or files == [""]:
            return f"No content matches found for pattern '{pattern}'"

        # Get context for each match
        output_lines = []
        for file_path in files[:20]:  # Limit to 20 files
            if not file_path:
                continue
            output_lines.append(f"\n=== {file_path} ===")

            # Get matching lines with context
            ctx_cmd = ["grep", "-n", "-B", "1", "-A", "1", pattern, file_path]
            ctx_result = subprocess.run(
                ctx_cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if ctx_result.stdout:
                output_lines.append(ctx_result.stdout.strip())

        if len(files) > 20:
            output_lines.append(f"\n... ({len(files)} total files with matches, showing first 20)")

        return "\n".join(output_lines)

    except subprocess.TimeoutExpired:
        return f"Search timed out while looking for '{pattern}'"
    except Exception as e:
        return f"Error during content search: {e}"
