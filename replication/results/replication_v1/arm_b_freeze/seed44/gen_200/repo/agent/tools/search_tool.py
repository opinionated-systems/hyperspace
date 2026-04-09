"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities.
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
            "Search for files and content. "
            "Commands: find_files, grep, find_and_replace. "
            "Supports regex patterns and file filtering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_and_replace"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex for grep, glob for find).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Filter files by pattern (e.g., '*.py').",
                },
                "replacement": {
                    "type": "string",
                    "description": "Replacement string for find_and_replace.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 50,
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path_allowed(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True, ""
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, ""


def _find_files(base_path: str, pattern: str, max_results: int) -> str:
    """Find files matching a glob pattern."""
    allowed, error = _check_path_allowed(base_path)
    if not allowed:
        return error
    
    p = Path(base_path)
    if not p.exists():
        return f"Error: path {base_path} does not exist"
    
    if not p.is_dir():
        return f"Error: {base_path} is not a directory"
    
    try:
        matches = list(p.rglob(pattern))
        matches = matches[:max_results]
        
        if not matches:
            return f"No files matching '{pattern}' found in {base_path}"
        
        result = [f"Found {len(matches)} file(s) matching '{pattern}':"]
        for m in matches:
            rel_path = m.relative_to(p)
            result.append(f"  {rel_path}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error searching files: {e}"


def _grep(base_path: str, pattern: str, file_pattern: str | None, max_results: int) -> str:
    """Search for pattern in file contents."""
    allowed, error = _check_path_allowed(base_path)
    if not allowed:
        return error
    
    p = Path(base_path)
    if not p.exists():
        return f"Error: path {base_path} does not exist"
    
    try:
        results = []
        count = 0
        
        # Determine which files to search
        if p.is_file():
            files_to_search = [p]
        else:
            if file_pattern:
                files_to_search = list(p.rglob(file_pattern))
            else:
                # Default: search common text file types
                files_to_search = []
                for ext in ["*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.js", "*.ts", "*.html", "*.css"]:
                    files_to_search.extend(p.rglob(ext))
        
        # Compile regex
        try:
            regex = re.compile(pattern, re.MULTILINE)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"
        
        for file_path in files_to_search:
            if not file_path.is_file():
                continue
            
            # Skip binary files and very large files
            try:
                size = file_path.stat().st_size
                if size > 10 * 1024 * 1024:  # Skip files > 10MB
                    continue
                
                content = file_path.read_text(errors="ignore")
            except (IOError, OSError, UnicodeDecodeError):
                continue
            
            matches = list(regex.finditer(content))
            if matches:
                rel_path = file_path.relative_to(p) if p.is_dir() else file_path.name
                for match in matches:
                    if count >= max_results:
                        break
                    
                    # Get context around match
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end].replace("\n", " ")
                    
                    results.append(f"{rel_path}:{match.start()}: {context}")
                    count += 1
                
                if count >= max_results:
                    break
        
        if not results:
            return f"No matches for pattern '{pattern}' found"
        
        header = f"Found {count} match(es) for pattern '{pattern}':"
        return header + "\n" + "\n".join(results)
    
    except Exception as e:
        return f"Error during grep: {e}"


def _find_and_replace(base_path: str, pattern: str, replacement: str, file_pattern: str | None) -> str:
    """Find and replace pattern in files."""
    allowed, error = _check_path_allowed(base_path)
    if not allowed:
        return error
    
    if replacement is None:
        return "Error: replacement string required for find_and_replace"
    
    p = Path(base_path)
    if not p.exists():
        return f"Error: path {base_path} does not exist"
    
    try:
        files_modified = 0
        total_replacements = 0
        
        # Determine which files to modify
        if p.is_file():
            files_to_modify = [p]
        else:
            if file_pattern:
                files_to_modify = list(p.rglob(file_pattern))
            else:
                return "Error: file_pattern required when path is a directory"
        
        # Compile regex
        try:
            regex = re.compile(pattern, re.MULTILINE)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"
        
        for file_path in files_to_modify:
            if not file_path.is_file():
                continue
            
            try:
                content = file_path.read_text()
            except (IOError, OSError, UnicodeDecodeError):
                continue
            
            new_content, num_replacements = regex.subn(replacement, content)
            
            if num_replacements > 0:
                file_path.write_text(new_content)
                files_modified += 1
                total_replacements += num_replacements
        
        return f"Modified {files_modified} file(s) with {total_replacements} replacement(s)"
    
    except Exception as e:
        return f"Error during find_and_replace: {e}"


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
    replacement: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    if not Path(path).is_absolute():
        return f"Error: {path} is not an absolute path"
    
    if command == "find_files":
        return _find_files(path, pattern, max_results)
    elif command == "grep":
        return _grep(path, pattern, file_pattern, max_results)
    elif command == "find_and_replace":
        return _find_and_replace(path, pattern, replacement or "", file_pattern)
    else:
        return f"Error: unknown command {command}"
