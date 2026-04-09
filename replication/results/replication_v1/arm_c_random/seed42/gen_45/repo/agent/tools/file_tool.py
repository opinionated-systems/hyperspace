"""
File search tool: grep and find files in the codebase.

Provides capabilities to search for patterns in files and list directory contents.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


def tool_info() -> dict:
    """Return tool specification for OpenAI function calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for files or content within files. Supports grep pattern matching and directory listing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["grep", "find", "list"],
                        "description": "Command to execute: 'grep' searches file contents, 'find' searches filenames, 'list' lists directory contents",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (required for grep and find)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to search in (default: current directory)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively (default: true for grep/find)",
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "File extension filter, e.g., '.py' (optional)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50)",
                    },
                },
                "required": ["command"],
            },
        },
    }


def tool_function(
    command: str,
    pattern: str | None = None,
    path: str | None = None,
    recursive: bool = True,
    file_extension: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """Execute file search command.

    Args:
        command: One of 'grep', 'find', 'list'
        pattern: Search pattern (for grep/find)
        path: Directory or file to search
        recursive: Whether to search recursively
        file_extension: Filter by file extension
        max_results: Maximum results to return

    Returns:
        Dict with results or error info.
    """
    if path is None:
        path = "."

    # Validate path exists
    if not os.path.exists(path):
        return {
            "error": f"Path does not exist: {path}",
            "results": [],
        }

    results: list[dict] = []

    try:
        if command == "list":
            # List directory contents
            if os.path.isfile(path):
                return {
                    "error": f"Path is a file, not a directory: {path}",
                    "results": [],
                }
            
            entries = os.listdir(path)
            for entry in sorted(entries)[:max_results]:
                full_path = os.path.join(path, entry)
                entry_type = "directory" if os.path.isdir(full_path) else "file"
                results.append({
                    "name": entry,
                    "type": entry_type,
                    "path": full_path,
                })

        elif command == "find":
            # Find files by name pattern
            if pattern is None:
                return {
                    "error": "Pattern is required for find command",
                    "results": [],
                }

            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            
            if os.path.isfile(path):
                # Single file check
                if compiled_pattern.search(os.path.basename(path)):
                    results.append({
                        "path": path,
                        "name": os.path.basename(path),
                    })
            else:
                # Directory search
                for root, dirs, files in os.walk(path):
                    for filename in files:
                        if file_extension and not filename.endswith(file_extension):
                            continue
                        if compiled_pattern.search(filename):
                            full_path = os.path.join(root, filename)
                            results.append({
                                "path": full_path,
                                "name": filename,
                            })
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
                    if not recursive:
                        break

        elif command == "grep":
            # Search file contents
            if pattern is None:
                return {
                    "error": "Pattern is required for grep command",
                    "results": [],
                }

            # Use ripgrep if available, fallback to Python implementation
            try:
                cmd = ["rg", "-n", "--json", pattern]
                if not recursive:
                    cmd.append("--max-depth=1")
                if file_extension:
                    cmd.extend(["-g", f"*{file_extension}"])
                cmd.append(path)
                
                output = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                for line in output.stdout.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        import json
                        data = json.loads(line)
                        if data.get("type") == "match":
                            match_data = data.get("data", {})
                            path_match = match_data.get("path", {}).get("text", "")
                            lines = match_data.get("lines", {}).get("text", "")
                            line_num = match_data.get("line_number", 0)
                            results.append({
                                "path": path_match,
                                "line_number": line_num,
                                "content": lines.strip(),
                            })
                            if len(results) >= max_results:
                                break
                    except json.JSONDecodeError:
                        continue
                        
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # Fallback to Python implementation
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                
                if os.path.isfile(path):
                    files_to_search = [path]
                else:
                    files_to_search = []
                    for root, dirs, files in os.walk(path):
                        for filename in files:
                            if file_extension and not filename.endswith(file_extension):
                                continue
                            files_to_search.append(os.path.join(root, filename))
                        if not recursive:
                            break
                
                for filepath in files_to_search:
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            for line_num, line in enumerate(f, 1):
                                if compiled_pattern.search(line):
                                    results.append({
                                        "path": filepath,
                                        "line_number": line_num,
                                        "content": line.strip(),
                                    })
                                    if len(results) >= max_results:
                                        break
                    except (IOError, OSError):
                        continue
                    if len(results) >= max_results:
                        break

        else:
            return {
                "error": f"Unknown command: {command}",
                "results": [],
            }

        return {
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        return {
            "error": f"Error executing {command}: {str(e)}",
            "results": [],
        }
