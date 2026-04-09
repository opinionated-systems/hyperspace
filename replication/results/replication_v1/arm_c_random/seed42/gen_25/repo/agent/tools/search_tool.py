"""
Search tool: search for patterns in files using grep with enhanced features.

Provides file content search capabilities with context lines, count-only mode,
and improved result formatting.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


# Statistics tracking
_search_stats = {
    "total_searches": 0,
    "total_matches": 0,
    "patterns_searched": set(),
}


def get_search_stats() -> dict:
    """Return search statistics."""
    return {
        "total_searches": _search_stats["total_searches"],
        "total_matches": _search_stats["total_matches"],
        "unique_patterns": len(_search_stats["patterns_searched"]),
    }


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep. "
            "Supports regex patterns, recursive search, context lines, and count-only mode."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (absolute path).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive (default: True).",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show before/after match (default: 0).",
                },
                "count_only": {
                    "type": "boolean",
                    "description": "Return only count of matches per file (default: False).",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = Path(root).resolve()


def _format_match_line(line: str, is_match: bool = True) -> str:
    """Format a match line with visual indicator."""
    if is_match:
        return line.replace(":", ":>", 1) if ":" in line else f"> {line}"
    return line


def _group_matches_by_file(lines: list[str]) -> dict[str, list[str]]:
    """Group match lines by file path."""
    grouped: dict[str, list[str]] = {}
    for line in lines:
        if ":" in line:
            file_path, rest = line.split(":", 1)
            if file_path not in grouped:
                grouped[file_path] = []
            grouped[file_path].append(rest)
        else:
            # Handle lines without proper format
            if "unknown" not in grouped:
                grouped["unknown"] = []
            grouped["unknown"].append(line)
    return grouped


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    case_sensitive: bool = True,
    context_lines: int = 0,
    count_only: bool = False,
) -> str:
    """Execute a search using grep with enhanced features.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        file_extension: Optional file extension filter
        case_sensitive: Whether search is case sensitive
        context_lines: Number of context lines to show (0 for none)
        count_only: Return only count per file
    
    Returns:
        Formatted search results with statistics
    """
    global _search_stats
    
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = Path(path).resolve()
            try:
                resolved.relative_to(_ALLOWED_ROOT)
            except ValueError:
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        # Update statistics
        _search_stats["total_searches"] += 1
        _search_stats["patterns_searched"].add(pattern)

        # Build grep command
        cmd = ["grep", "-r" if p.is_dir() else "", "-n"]
        
        if not case_sensitive:
            cmd.append("-i")
        
        # Add context lines if requested
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        
        # Add count-only flag if requested
        if count_only:
            cmd.append("-c")
        
        # Add pattern
        cmd.append(pattern)
        
        # Add path
        cmd.append(str(p))
        
        # Filter by extension if specified
        if file_extension and p.is_dir():
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Exclude common non-source directories
        exclude_dirs = [
            "__pycache__", "node_modules", ".git", ".svn",
            "dist", "build", ".pytest_cache", ".mypy_cache"
        ]
        for d in exclude_dirs:
            cmd.extend(["--exclude-dir", d])
        
        # Remove empty strings from cmd
        cmd = [c for c in cmd if c]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            _search_stats["total_matches"] += len(lines)
            
            if count_only:
                # Format count-only results
                counts = []
                for line in lines:
                    if ":" in line:
                        file_path, count = line.rsplit(":", 1)
                        try:
                            if int(count) > 0:
                                counts.append((file_path, int(count)))
                        except ValueError:
                            pass
                
                if not counts:
                    return f"No matches found for pattern '{pattern}' in {path}"
                
                total = sum(c for _, c in counts)
                output = [f"Match counts for '{pattern}':", "-" * 40]
                for file_path, count in sorted(counts, key=lambda x: -x[1]):
                    output.append(f"{count:4d}: {file_path}")
                output.append("-" * 40)
                output.append(f"Total: {total} matches in {len(counts)} files")
                return "\n".join(output)
            
            # Full results mode
            if len(lines) > 50:
                output_lines = lines[:50]
                truncated = True
            else:
                output_lines = lines
                truncated = False
            
            # Group by file for better readability
            grouped = _group_matches_by_file(output_lines)
            
            formatted = [f"Search results for '{pattern}':", "=" * 50]
            for file_path, matches in grouped.items():
                rel_path = file_path.replace(str(p), ".", 1) if str(p) in file_path else file_path
                formatted.append(f"\n{rel_path}")
                formatted.append("-" * min(40, len(rel_path)))
                for match in matches:
                    formatted.append(_format_match_line(match))
            
            if truncated:
                formatted.append(f"\n... and {len(lines) - 50} more matches")
            
            formatted.append(f"\n{'=' * 50}")
            formatted.append(f"Found {len(lines)} total matches")
            
            return "\n".join(formatted)
            
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            error_msg = result.stderr.strip()
            if "invalid regex" in error_msg.lower() or "bad regex" in error_msg.lower():
                return f"Error: Invalid regex pattern '{pattern}'. Please check your syntax."
            return f"Error: {error_msg}"
            
    except Exception as e:
        return f"Error: {e}"
