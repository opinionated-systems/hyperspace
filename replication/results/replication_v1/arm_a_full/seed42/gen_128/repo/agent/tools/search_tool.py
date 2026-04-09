"""
Search tool: find files by name or content within the repository.

Provides grep-like functionality to search for patterns in file contents
and find files by name patterns.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files by name pattern or content within the repository. "
            "Useful for finding where specific functions, classes, or patterns are defined."
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
                    "description": "Maximum number of results to return (default: 50).",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show for content matches (default: 0).",
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


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
    context_lines: int = 0,
) -> str:
    """Search for files by name or content.
    
    Args:
        pattern: Search pattern (regex for content, glob for filename)
        search_type: "content" or "filename"
        path: Directory to search in (default: allowed root)
        file_extension: Optional extension filter (e.g., ".py")
        max_results: Maximum results to return
        context_lines: Number of context lines for content matches
    
    Returns:
        Formatted search results
    """
    search_path = path or _ALLOWED_ROOT or "."
    
    if not _is_within_allowed(search_path):
        return f"Error: Search path '{search_path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist."
    
    results: list[str] = []
    
    if search_type == "filename":
        # Use find command for filename search
        cmd = ["find", search_path, "-type", "f", "-name", pattern]
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            
            if file_extension:
                files = [f for f in files if f.endswith(file_extension)]
            
            results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out."
        except Exception as e:
            return f"Error during search: {e}"
    
    elif search_type == "content":
        # Use grep for content search with optional context
        include_pattern = f"*{file_extension}" if file_extension else "*"
        
        if context_lines > 0:
            # Use grep with context lines
            cmd = [
                "grep", "-r", "-n", "-C", str(context_lines),
                "--include", include_pattern,
                pattern, search_path
            ]
        else:
            # Use grep without context (just list files)
            cmd = [
                "grep", "-r", "-n", "-l", "--include", include_pattern,
                pattern, search_path
            ]
        
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            
            if context_lines > 0:
                # Parse grep output with context
                lines = output.stdout.strip().split("\n") if output.stdout.strip() else []
                results = lines[:max_results * (2 * context_lines + 1)]
            else:
                files = output.stdout.strip().split("\n") if output.stdout.strip() else []
                files = [f for f in files if f and _is_within_allowed(f)]
                results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out."
        except Exception as e:
            return f"Error during search: {e}"
    
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
        if _ALLOWED_ROOT and isinstance(result, str) and result.startswith(_ALLOWED_ROOT):
            rel_path = os.path.relpath(result, _ALLOWED_ROOT)
            output_lines.append(f"{i}. {rel_path}")
        else:
            output_lines.append(f"{i}. {result}")
    
    if len(results) == max_results:
        output_lines.append("")
        output_lines.append(f"(Results limited to {max_results}. Refine your search for more specific results.)")
    
    return "\n".join(output_lines)


def search_with_preview(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 10,
    preview_lines: int = 3,
) -> str:
    """Search for content and show a preview of each match.
    
    This is useful for quickly understanding what each match contains
    without having to open each file individually.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory to search in
        file_extension: Optional extension filter
        max_results: Maximum number of files to show
        preview_lines: Number of lines to show around each match
    
    Returns:
        Formatted search results with previews
    """
    search_path = path or _ALLOWED_ROOT or "."
    
    if not _is_within_allowed(search_path):
        return f"Error: Search path '{search_path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist."
    
    include_pattern = f"*{file_extension}" if file_extension else "*"
    
    # Use grep with context lines for preview
    cmd = [
        "grep", "-r", "-n", "-C", str(preview_lines),
        "--include", include_pattern,
        pattern, search_path
    ]
    
    try:
        output = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        
        if not output.stdout.strip():
            return f"No matches found for pattern '{pattern}'."
        
        # Group results by file
        lines = output.stdout.strip().split("\n")
        current_file = None
        file_matches: dict[str, list[str]] = {}
        
        for line in lines:
            if line.startswith("--"):
                # Separator between files
                continue
            
            # Parse line to extract file path
            match = re.match(r"^([^:]+):(\d+):(.*)$", line)
            if match:
                file_path = match.group(1)
                if file_path not in file_matches:
                    file_matches[file_path] = []
                file_matches[file_path].append(line)
        
        # Format output
        output_lines = [
            f"Found matches in {len(file_matches)} file(s) for pattern '{pattern}':",
            "",
        ]
        
        for i, (file_path, matches) in enumerate(list(file_matches.items())[:max_results], 1):
            rel_path = os.path.relpath(file_path, _ALLOWED_ROOT) if _ALLOWED_ROOT else file_path
            output_lines.append(f"{i}. {rel_path}")
            for match_line in matches[:5]:  # Limit matches per file
                output_lines.append(f"   {match_line}")
            if len(matches) > 5:
                output_lines.append(f"   ... and {len(matches) - 5} more matches")
            output_lines.append("")
        
        if len(file_matches) > max_results:
            output_lines.append(f"(Results limited to {max_results} files. Refine your search for more specific results.)")
        
        return "\n".join(output_lines)
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out."
    except Exception as e:
        return f"Error during search: {e}"
