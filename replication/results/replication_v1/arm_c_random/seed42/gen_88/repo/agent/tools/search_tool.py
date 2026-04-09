"""Search tool for finding files by name or content pattern.

Provides file search capabilities to complement bash and editor tools.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _find_files_by_name(
    root: str,
    pattern: str,
    max_results: int = 100,
) -> list[str]:
    """Find files matching a glob pattern.

    Args:
        root: Directory to search in
        pattern: Glob pattern (e.g., "*.py", "**/*.txt")
        max_results: Maximum number of results to return

    Returns:
        List of matching file paths (relative to root)
    """
    results = []
    root_path = Path(root).resolve()

    for path in root_path.rglob(pattern):
        if path.is_file():
            rel_path = path.relative_to(root_path)
            results.append(str(rel_path))
            if len(results) >= max_results:
                break

    return results


def _find_files_by_content(
    root: str,
    pattern: str,
    file_pattern: str = "*",
    max_results: int = 50,
    max_file_size: int = 1024 * 1024,  # 1MB
) -> list[dict]:
    """Find files containing a regex pattern.

    Args:
        root: Directory to search in
        pattern: Regex pattern to search for
        file_pattern: Glob pattern to filter files
        max_results: Maximum number of matching files to return
        max_file_size: Skip files larger than this

    Returns:
        List of dicts with file path and matching lines
    """
    results = []
    root_path = Path(root).resolve()
    regex = re.compile(pattern)

    for path in root_path.rglob(file_pattern):
        if not path.is_file():
            continue
        if path.stat().st_size > max_file_size:
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if regex.search(content):
                rel_path = path.relative_to(root_path)
                # Get first few matching lines
                matches = []
                for i, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        matches.append({"line": i, "text": line[:200]})
                        if len(matches) >= 3:
                            break

                results.append({
                    "path": str(rel_path),
                    "matches": matches,
                })

                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.debug("Error reading %s: %s", path, e)
            continue

    return results


def tool_function(
    command: str,
    root: str = ".",
    pattern: str = "",
    file_pattern: str = "*",
    max_results: int = 50,
) -> dict[str, Any]:
    """Search for files by name or content.

    Args:
        command: Type of search - "name" (glob pattern) or "content" (regex)
        root: Directory to search in (default: current directory)
        pattern: Search pattern (glob for "name", regex for "content")
        file_pattern: When using "content", filter files by this glob
        max_results: Maximum number of results

    Returns:
        Dict with search results
    """
    if not pattern:
        return {
            "success": False,
            "error": "pattern is required",
            "results": [],
        }

    root = os.path.abspath(root)

    if not os.path.isdir(root):
        return {
            "success": False,
            "error": f"Directory not found: {root}",
            "results": [],
        }

    try:
        if command == "name":
            results = _find_files_by_name(root, pattern, max_results)
            return {
                "success": True,
                "command": command,
                "pattern": pattern,
                "root": root,
                "count": len(results),
                "results": results,
            }
        elif command == "content":
            results = _find_files_by_content(
                root, pattern, file_pattern, max_results
            )
            return {
                "success": True,
                "command": command,
                "pattern": pattern,
                "file_pattern": file_pattern,
                "root": root,
                "count": len(results),
                "results": results,
            }
        else:
            return {
                "success": False,
                "error": f"Unknown command: {command}. Use 'name' or 'content'.",
                "results": [],
            }
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "results": [],
        }
    except Exception as e:
        logger.exception("Search failed")
        return {
            "success": False,
            "error": str(e),
            "results": [],
        }


def tool_info() -> dict:
    """Return the tool specification for LLM tool calling."""
    return {
        "name": "search",
        "description": (
            "Search for files by name (glob pattern) or content (regex pattern). "
            "Use 'name' to find files matching a glob like '*.py' or '**/*.json'. "
            "Use 'content' to find files containing a regex pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["name", "content"],
                    "description": "Type of search: 'name' for glob pattern, 'content' for regex pattern",
                },
                "root": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern - glob for 'name', regex for 'content'",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "When using 'content', only search files matching this glob (default: '*')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                },
            },
            "required": ["command", "pattern"],
        },
    }


def get_tool_spec() -> dict:
    """Return the tool specification for LLM tool calling (OpenAI format)."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search for files by name (glob pattern) or content (regex pattern). "
                "Use 'name' to find files matching a glob like '*.py' or '**/*.json'. "
                "Use 'content' to find files containing a regex pattern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["name", "content"],
                        "description": "Type of search: 'name' for glob pattern, 'content' for regex pattern",
                    },
                    "root": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern - glob for 'name', regex for 'content'",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "When using 'content', only search files matching this glob (default: '*')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50)",
                    },
                },
                "required": ["command", "pattern"],
            },
        },
    }
