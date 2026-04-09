"""
Search tool: find files by name or content within the repository.

Provides grep-like functionality to search for patterns in file contents
and find files by name patterns.
Enhanced with better error handling and input validation.
"""

from __future__ import annotations

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum results to prevent memory issues
MAX_RESULTS_LIMIT = 1000


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files by name pattern or content within the repository. "
            "Useful for finding where specific functions, classes, or patterns are defined. "
            "Results are limited to 1000 matches maximum."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex for content, glob for filename).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: repo root).",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "Whether to search in file contents or filenames.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50, max: 1000).",
                },
            },
            "required": ["pattern", "search_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for searches."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def _validate_search_params(
    pattern: str,
    search_type: str,
    max_results: int,
) -> tuple[bool, str]:
    """Validate search parameters.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(pattern, str) or not pattern.strip():
        return False, "Pattern must be a non-empty string"
    
    if search_type not in ("content", "filename"):
        return False, f"search_type must be 'content' or 'filename', got '{search_type}'"
    
    if not isinstance(max_results, int) or max_results < 1:
        return False, f"max_results must be a positive integer, got {max_results}"
    
    if max_results > MAX_RESULTS_LIMIT:
        return False, f"max_results cannot exceed {MAX_RESULTS_LIMIT}, got {max_results}"
    
    return True, ""


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for files by name or content.
    
    Args:
        pattern: Search pattern (regex for content, glob for filename)
        search_type: "content" or "filename"
        path: Directory to search in (default: allowed root)
        file_extension: Optional extension filter (e.g., ".py")
        max_results: Maximum results to return
    
    Returns:
        Formatted search results
    """
    # Validate parameters
    is_valid, error_msg = _validate_search_params(pattern, search_type, max_results)
    if not is_valid:
        return f"Error: {error_msg}"
    
    search_path = path or _ALLOWED_ROOT or "."
    
    if not isinstance(search_path, str):
        return f"Error: Path must be a string, got {type(search_path).__name__}"
    
    if not _is_within_allowed(search_path):
        logger.warning(f"Search path '{search_path}' outside allowed root '{_ALLOWED_ROOT}'")
        return f"Error: Search path '{search_path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist."
    
    if not os.path.isdir(search_path):
        return f"Error: Path '{search_path}' is not a directory."
    
    results: list[str] = []
    
    if search_type == "filename":
        # Use find command for filename search
        cmd = ["find", search_path, "-type", "f", "-name", pattern]
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if output.returncode != 0 and output.stderr:
                logger.warning(f"find command stderr: {output.stderr}")
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            
            if file_extension:
                files = [f for f in files if f.endswith(file_extension)]
            
            results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds."
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error in filename search: {e}")
            return f"Error: Subprocess failed - {e}"
        except Exception as e:
            logger.error(f"Unexpected error in filename search: {e}")
            return f"Error during search: {type(e).__name__}: {e}"
    
    elif search_type == "content":
        # Use grep for content search
        # Build grep command with file extension filter if provided
        include_pattern = f"*{file_extension}" if file_extension else "*"
        
        cmd = [
            "grep", "-r", "-n", "-l", "--include", include_pattern,
            pattern, search_path
        ]
        
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            # grep returns 1 when no matches found, which is not an error
            if output.returncode not in (0, 1):
                logger.warning(f"grep command returned {output.returncode}: {output.stderr}")
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds."
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error in content search: {e}")
            return f"Error: Subprocess failed - {e}"
        except Exception as e:
            logger.error(f"Unexpected error in content search: {e}")
            return f"Error during search: {type(e).__name__}: {e}"
    
    else:
        return f"Error: Invalid search_type '{search_type}'. Use 'content' or 'filename'."
    
    if not results:
        return f"No results found for pattern '{pattern}' (type: {search_type})."
    
    # Format results
    output_lines = [
        f"Found {len(results)} result(s) for pattern '{pattern}' (type: {search_type}):",
        "",
    ]
    
    for i, result in enumerate(results, 1):
        # Show relative path if within allowed root
        if _ALLOWED_ROOT and result.startswith(_ALLOWED_ROOT):
            rel_path = os.path.relpath(result, _ALLOWED_ROOT)
            output_lines.append(f"{i}. {rel_path}")
        else:
            output_lines.append(f"{i}. {result}")
    
    if len(results) == max_results:
        output_lines.append("")
        output_lines.append(f"(Results limited to {max_results}. Refine your search for more specific results.)")
    
    return "\n".join(output_lines)
