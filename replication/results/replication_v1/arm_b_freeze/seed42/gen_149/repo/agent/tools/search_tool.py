"""
Search tool: search for files and content within the codebase.

Provides grep-like and find-like functionality for the agent.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files or content within a directory. Supports grep-like text search and find-like file discovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["grep", "find", "file_regex"],
                        "description": "Search command: 'grep' for content search, 'find' for file listing, 'file_regex' for filename pattern matching",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (text for grep, glob for find, regex for file_regex)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively (default: true)",
                        "default": True,
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case sensitive search for grep (default: false)",
                        "default": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50)",
                        "default": 50,
                    },
                },
                "required": ["command", "path", "pattern"],
            },
        },
    }


def _grep_search(path: str, pattern: str, recursive: bool, case_sensitive: bool, max_results: int) -> dict:
    """Search for text content within files using grep."""
    flags = ["-n"]  # Line numbers
    if not case_sensitive:
        flags.append("-i")  # Case insensitive
    if recursive:
        flags.append("-r")
    
    cmd = ["grep"] + flags + [pattern]
    if not recursive:
        cmd.append("*")
    else:
        cmd.append(".")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        lines = [l for l in lines if l][:max_results]
        
        return {
            "success": True,
            "matches": lines,
            "count": len(lines),
            "truncated": len(lines) >= max_results,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Search timed out (30s limit)",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def _find_files(path: str, pattern: str, recursive: bool, max_results: int) -> dict:
    """List files matching a glob pattern."""
    import fnmatch
    
    matches = []
    
    if recursive:
        for root, dirs, files in os.walk(path):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    matches.append(os.path.join(root, filename))
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break
    else:
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath) and fnmatch.fnmatch(filename, pattern):
                matches.append(filepath)
                if len(matches) >= max_results:
                    break
    
    return {
        "success": True,
        "matches": matches,
        "count": len(matches),
        "truncated": len(matches) >= max_results,
    }


def _file_regex_search(path: str, pattern: str, recursive: bool, max_results: int) -> dict:
    """Search for files matching a regex pattern."""
    matches = []
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
        }
    
    if recursive:
        for root, dirs, files in os.walk(path):
            for filename in files:
                if regex.search(filename):
                    matches.append(os.path.join(root, filename))
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break
    else:
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath) and regex.search(filename):
                matches.append(filepath)
                if len(matches) >= max_results:
                    break
    
    return {
        "success": True,
        "matches": matches,
        "count": len(matches),
        "truncated": len(matches) >= max_results,
    }


def tool_function(
    command: str,
    path: str,
    pattern: str,
    recursive: bool = True,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> dict:
    """Execute search command.
    
    Args:
        command: 'grep', 'find', or 'file_regex'
        path: Directory to search in
        pattern: Search pattern
        recursive: Search recursively
        case_sensitive: Case sensitive for grep
        max_results: Maximum results to return
    
    Returns:
        Dict with success status and matches
    """
    # Validate path exists
    if not os.path.exists(path):
        return {
            "success": False,
            "error": f"Path does not exist: {path}",
        }
    
    if not os.path.isdir(path):
        return {
            "success": False,
            "error": f"Path is not a directory: {path}",
        }
    
    if command == "grep":
        return _grep_search(path, pattern, recursive, case_sensitive, max_results)
    elif command == "find":
        return _find_files(path, pattern, recursive, max_results)
    elif command == "file_regex":
        return _file_regex_search(path, pattern, recursive, max_results)
    else:
        return {
            "success": False,
            "error": f"Unknown command: {command}",
        }
