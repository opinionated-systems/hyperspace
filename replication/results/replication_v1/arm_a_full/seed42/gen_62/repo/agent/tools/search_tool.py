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
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive (default: false).",
                },
                "show_context": {
                    "type": "boolean",
                    "description": "For content search, show matching line context (default: true).",
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
    case_sensitive: bool = False,
    show_context: bool = True,
) -> str:
    """Search for files by name or content.
    
    Args:
        pattern: Search pattern (regex for content, glob for filename)
        search_type: "content" or "filename"
        path: Directory to search in (default: allowed root)
        file_extension: Optional extension filter (e.g., ".py")
        max_results: Maximum results to return (1-100)
        case_sensitive: Whether search is case-sensitive (default: False)
        show_context: For content search, show matching line context (default: True)
    
    Returns:
        Formatted search results
    """
    # Validate inputs
    if not pattern or not isinstance(pattern, str):
        return "Error: pattern must be a non-empty string"
    
    if search_type not in ("content", "filename"):
        return f"Error: Invalid search_type '{search_type}'. Use 'content' or 'filename'."
    
    # Clamp max_results to reasonable range
    if not isinstance(max_results, int):
        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            max_results = 50
    max_results = max(1, min(100, max_results))
    
    search_path = path or _ALLOWED_ROOT or "."
    
    if not _is_within_allowed(search_path):
        return f"Error: Search path '{search_path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist."
    
    if not os.path.isdir(search_path):
        return f"Error: Path '{search_path}' is not a directory."
    
    results: list[str] = []
    context_lines: dict[str, list[str]] = {}
    
    if search_type == "filename":
        # Use find command for filename search
        # For case-insensitive filename search, use -iname
        name_flag = "-iname" if not case_sensitive else "-name"
        cmd = ["find", search_path, "-type", "f", name_flag, pattern]
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if output.returncode != 0 and output.stderr:
                # find returns non-zero for some patterns but still gives results
                pass
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            
            if file_extension:
                ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
                files = [f for f in files if f.endswith(ext)]
            
            results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
        except FileNotFoundError:
            return "Error: 'find' command not available."
        except Exception as e:
            return f"Error during filename search: {type(e).__name__}: {e}"
    
    elif search_type == "content":
        # Use grep for content search
        # Build grep command with file extension filter if provided
        include_pattern = f"*{file_extension}" if file_extension else "*"
        
        # Add case-insensitive flag if needed
        case_flag = "-i" if not case_sensitive else ""
        
        if show_context:
            # Show context lines around matches
            cmd = [
                "grep", "-r", "-n", "-C", "2", "-l" if not show_context else "",
            ]
            if case_flag:
                cmd.append(case_flag)
            cmd.extend(["--include", include_pattern, pattern, search_path])
            # Remove empty strings
            cmd = [c for c in cmd if c]
        else:
            cmd = [
                "grep", "-r", "-n", "-l",
            ]
            if case_flag:
                cmd.append(case_flag)
            cmd.extend(["--include", include_pattern, pattern, search_path])
        
        try:
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            # grep returns 1 when no matches found, which is OK
            if show_context:
                # Parse context output
                current_file = None
                for line in output.stdout.split("\n"):
                    if line.startswith("--"):
                        continue
                    if ":" in line:
                        parts = line.split(":", 1)
                        file_path = parts[0]
                        if file_path and _is_within_allowed(file_path):
                            if file_path not in results:
                                if len(results) < max_results:
                                    results.append(file_path)
                                    context_lines[file_path] = []
                            if file_path in context_lines:
                                context_lines[file_path].append(line)
            else:
                files = output.stdout.strip().split("\n") if output.stdout.strip() else []
                files = [f for f in files if f and _is_within_allowed(f)]
                results = files[:max_results]
            
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds. Try a more specific pattern or narrower path."
        except FileNotFoundError:
            return "Error: 'grep' command not available."
        except Exception as e:
            return f"Error during content search: {type(e).__name__}: {e}"
    
    if not results:
        search_desc = f"pattern '{pattern}'"
        if file_extension:
            search_desc += f" with extension '{file_extension}'"
        case_desc = "case-sensitive" if case_sensitive else "case-insensitive"
        return f"No results found for {search_desc} (type: {search_type}, {case_desc})."
    
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
        
        # Add context lines if available
        if show_context and result in context_lines and context_lines[result]:
            output_lines.append("   Context:")
            for ctx_line in context_lines[result][:5]:  # Limit context lines
                output_lines.append(f"   {ctx_line}")
            if len(context_lines[result]) > 5:
                output_lines.append(f"   ... ({len(context_lines[result]) - 5} more matches)")
    
    if len(results) == max_results:
        output_lines.append("")
        output_lines.append(f"(Results limited to {max_results}. Refine your search for more specific results.)")
    
    return "\n".join(output_lines)
