"""
Search tool: find files by name or content within the repository.

Provides grep-like functionality to search for patterns in file contents
and find files by name patterns.
Enhanced with better error handling, logging, performance optimizations, and result ranking.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum results to prevent memory issues
HARD_MAX_RESULTS = 200
# Default timeout for searches (seconds)
SEARCH_TIMEOUT = 30
# Directories to exclude from search
EXCLUDE_DIRS = [
    "__pycache__", ".git", ".hg", ".svn", "node_modules",
    ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache",
    "dist", "build", ".eggs", "*.egg-info", ".idea", ".vscode"
]

# Priority keywords for ranking search results (higher = more important)
PRIORITY_KEYWORDS = {
    # High priority: definitions and core structures
    "def ": 10,
    "class ": 10,
    "import ": 8,
    "from ": 8,
    "__init__": 9,
    # Medium priority: common patterns
    "return": 5,
    "if ": 3,
    "for ": 3,
    "while ": 3,
    "try:": 4,
    "except": 4,
    "raise": 5,
    # Documentation
    '"""': 4,
    "'''": 4,
    "# ": 2,
}


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files by name pattern or content within the repository. "
            "Useful for finding where specific functions, classes, or patterns are defined. "
            "Uses system grep/find for performance. Max results: 200."
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
                    "description": f"Maximum number of results to return (default: 50, max: {HARD_MAX_RESULTS}).",
                    "minimum": 1,
                    "maximum": HARD_MAX_RESULTS,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case-sensitive (default: True).",
                    "default": True,
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
    logger.info(f"Search allowed root set to: {_ALLOWED_ROOT}")


def _is_within_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def _score_match(content: str, pattern: str) -> int:
    """Score a match based on content relevance.
    
    Higher scores indicate more important matches (definitions, imports, etc.)
    """
    score = 0
    content_lower = content.lower()
    pattern_lower = pattern.lower()
    
    # Base score for exact match
    if pattern_lower in content_lower:
        score += 5
    
    # Bonus for being at start of line (likely a definition)
    stripped = content.lstrip()
    if stripped.lower().startswith(pattern_lower):
        score += 10
    
    # Bonus for priority keywords in the line
    for keyword, weight in PRIORITY_KEYWORDS.items():
        if keyword in content:
            score += weight
    
    # Bonus for being a short line (more focused)
    if len(content) < 80:
        score += 3
    
    # Penalty for being in a comment (less important)
    if content.strip().startswith("#"):
        score -= 2
    
    return max(0, score)


def _rank_content_results(results: list[tuple[str, list[str]]], pattern: str) -> list[tuple[str, list[str]]]:
    """Rank content search results by relevance.
    
    Args:
        results: List of (filepath, matches) tuples
        pattern: The search pattern used
    
    Returns:
        Sorted list with highest scoring matches first
    """
    scored_results = []
    
    for filepath, matches in results:
        # Score each match and keep top ones
        scored_matches = []
        for match in matches:
            # Extract the actual content from "Line X: content" format
            if ": " in match:
                content = match.split(": ", 1)[1]
            else:
                content = match
            score = _score_match(content, pattern)
            scored_matches.append((score, match))
        
        # Sort matches by score (descending)
        scored_matches.sort(reverse=True)
        
        # Calculate file score (sum of top 3 match scores)
        file_score = sum(s for s, _ in scored_matches[:3])
        
        # Keep matches in scored order
        sorted_matches = [m for _, m in scored_matches]
        scored_results.append((file_score, filepath, sorted_matches))
    
    # Sort files by score (descending)
    scored_results.sort(reverse=True)
    
    # Return sorted results without scores
    return [(filepath, matches) for _, filepath, matches in scored_results]


def _build_find_cmd(search_path: str, pattern: str, file_extension: str | None) -> list[str]:
    """Build find command with exclusions."""
    cmd = ["find", search_path, "-type", "f", "-name", pattern]
    
    # Add exclusions
    for exclude in EXCLUDE_DIRS:
        cmd.extend(["-not", "-path", f"*/{exclude}/*"])
    
    return cmd


def _build_grep_cmd(search_path: str, pattern: str, file_extension: str | None, with_context: bool = False, case_sensitive: bool = True) -> list[str]:
    """Build grep command with exclusions.
    
    Args:
        search_path: Directory to search
        pattern: Search pattern
        file_extension: Optional file extension filter
        with_context: If True, include context lines (2 before, 2 after)
        case_sensitive: If False, use case-insensitive search
    """
    include_pattern = f"*{file_extension}" if file_extension else "*"
    
    cmd = [
        "grep", "-r",  # recursive
        "-I",  # Ignore binary files
        "--include", include_pattern,
    ]
    
    # Add case-insensitive flag if needed
    if not case_sensitive:
        cmd.append("-i")
    
    if with_context:
        cmd.extend(["-n", "-B", "2", "-A", "2"])  # line numbers + context
    else:
        cmd.append("-l")  # list filenames only
    
    # Add exclude patterns
    for exclude in EXCLUDE_DIRS:
        cmd.extend(["--exclude-dir", exclude])
    
    cmd.extend([pattern, search_path])
    
    return cmd


def _get_file_preview(filepath: str, max_lines: int = 5) -> str:
    """Get a preview of file content (first few lines).
    
    Args:
        filepath: Path to the file
        max_lines: Maximum number of lines to preview
        
    Returns:
        Preview text or error message
    """
    try:
        # Check if file is binary
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\x00' in chunk:
                return "[Binary file]"
        
        # Read first few lines
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
            
            if not lines:
                return "[Empty file]"
            
            preview = '\n'.join(lines)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            
            return preview
    except Exception as e:
        return f"[Unable to preview: {type(e).__name__}]"


def tool_function(
    pattern: str,
    search_type: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
    case_sensitive: bool = True,
) -> str:
    """Search for files by name or content.
    
    Args:
        pattern: Search pattern (regex for content, glob for filename)
        search_type: "content" or "filename"
        path: Directory to search in (default: allowed root)
        file_extension: Optional extension filter (e.g., ".py")
        max_results: Maximum results to return
        case_sensitive: Whether search should be case-sensitive (default: True)
    
    Returns:
        Formatted search results
    """
    # Validate inputs
    if not pattern:
        return "Error: pattern is required"
    
    if search_type not in ("content", "filename"):
        return f"Error: Invalid search_type '{search_type}'. Use 'content' or 'filename'."
    
    # Clamp max_results
    max_results = max(1, min(max_results, HARD_MAX_RESULTS))
    
    search_path = path or _ALLOWED_ROOT or "."
    search_path = os.path.abspath(search_path)
    
    if not _is_within_allowed(search_path):
        logger.warning(f"Search path '{search_path}' is outside allowed root '{_ALLOWED_ROOT}'")
        return f"Error: Search path '{search_path}' is outside allowed root."
    
    if not os.path.exists(search_path):
        return f"Error: Path '{search_path}' does not exist."
    
    if not os.path.isdir(search_path):
        return f"Error: Path '{search_path}' is not a directory."
    
    logger.info(f"Searching {search_type} for '{pattern}' in {search_path} (case_sensitive={case_sensitive})")
    
    results: list[str] = []
    
    try:
        if search_type == "filename":
            cmd = _build_find_cmd(search_path, pattern, file_extension)
            
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=SEARCH_TIMEOUT
            )
            
            if output.returncode != 0 and output.stderr:
                logger.warning(f"Find command stderr: {output.stderr}")
            
            files = output.stdout.strip().split("\n") if output.stdout.strip() else []
            files = [f for f in files if f and _is_within_allowed(f)]
            
            # Additional extension filter (find's -name doesn't handle extensions well with wildcards)
            if file_extension:
                files = [f for f in files if f.endswith(file_extension)]
            
            results = files[:max_results]
            
        elif search_type == "content":
            # Use context mode for better results (shows line numbers and context)
            cmd = _build_grep_cmd(search_path, pattern, file_extension, with_context=True, case_sensitive=case_sensitive)
            
            output = subprocess.run(
                cmd, capture_output=True, text=True, timeout=SEARCH_TIMEOUT
            )
            
            # grep returns 1 when no matches found, which is not an error
            if output.returncode not in (0, 1) and output.stderr:
                logger.warning(f"Grep command stderr: {output.stderr}")
            
            # Parse grep output with context
            content_results = []
            if output.stdout.strip():
                lines = output.stdout.strip().split("\n")
                current_file = None
                current_matches = []
                
                for line in lines:
                    if line.startswith("--"):
                        # Separator between files
                        if current_file and current_matches:
                            content_results.append((current_file, current_matches))
                        current_file = None
                        current_matches = []
                    elif ":" in line and not line.startswith("Binary"):
                        # This is a match line with format: filename:line_num:content
                        parts = line.split(":", 2)
                        if len(parts) >= 2:
                            # Extract filename and content
                            if current_file is None or (len(parts) == 3 and parts[0] != current_file):
                                if current_file and current_matches:
                                    content_results.append((current_file, current_matches))
                                current_file = parts[0] if len(parts) == 3 else "unknown"
                                current_matches = []
                            if len(parts) == 3:
                                line_num = parts[1]
                                content = parts[2]
                                current_matches.append(f"Line {line_num}: {content}")
                
                # Don't forget the last file
                if current_file and current_matches:
                    content_results.append((current_file, current_matches))
            
            # Filter by allowed root and rank by relevance
            content_results = [
                (f, matches) for f, matches in content_results 
                if f and _is_within_allowed(f)
            ]
            
            # Rank results by relevance before limiting
            content_results = _rank_content_results(content_results, pattern)
            
            # Limit results
            results = content_results[:max_results]
    
    except subprocess.TimeoutExpired:
        logger.warning(f"Search timed out after {SEARCH_TIMEOUT}s")
        return f"Error: Search timed out after {SEARCH_TIMEOUT} seconds. Try a more specific pattern or smaller directory."
    
    except FileNotFoundError as e:
        logger.error(f"Command not found: {e}")
        return f"Error: Required command not found: {e}. Ensure find/grep are installed."
    
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        return f"Error during search: {type(e).__name__}: {e}"
    
    if not results:
        return f"No results found for pattern '{pattern}' (type: {search_type})."
    
    # Format results
    output_lines = [
        f"Found {len(results)} result(s) for pattern '{pattern}' (type: {search_type}):",
        "",
    ]
    
    if search_type == "content":
        # Format content search results with context
        for i, (filepath, matches) in enumerate(results, 1):
            # Show relative path if within allowed root
            if _ALLOWED_ROOT and filepath.startswith(_ALLOWED_ROOT):
                rel_path = os.path.relpath(filepath, _ALLOWED_ROOT)
                output_lines.append(f"{i}. {rel_path}")
            else:
                output_lines.append(f"{i}. {filepath}")
            
            # Add match context (limit to first 5 matches per file)
            for match in matches[:5]:
                output_lines.append(f"   {match}")
            if len(matches) > 5:
                output_lines.append(f"   ... and {len(matches) - 5} more matches")
            output_lines.append("")
    else:
        # Format filename search results with previews
        for i, result in enumerate(results, 1):
            # Show relative path if within allowed root
            if _ALLOWED_ROOT and result.startswith(_ALLOWED_ROOT):
                rel_path = os.path.relpath(result, _ALLOWED_ROOT)
                output_lines.append(f"{i}. {rel_path}")
            else:
                output_lines.append(f"{i}. {result}")
            
            # Add file preview for text files
            preview = _get_file_preview(result, max_lines=3)
            if preview:
                # Indent preview lines
                preview_lines = preview.split('\n')
                for line in preview_lines:
                    output_lines.append(f"   {line}")
                output_lines.append("")
    
    if len(results) == max_results:
        output_lines.append("")
        output_lines.append(f"(Results limited to {max_results}. Refine your search for more specific results.)")
    
    logger.info(f"Search completed: {len(results)} results")
    return "\n".join(output_lines)
