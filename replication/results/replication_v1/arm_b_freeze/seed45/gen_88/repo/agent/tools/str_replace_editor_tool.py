"""
Advanced string replacement editor tool with preview and multi-replace capabilities.

Extends the basic editor tool with:
- Preview mode: see changes before applying
- Multi-replace: replace multiple occurrences with confirmation
- Regex support: pattern-based replacements
- Context display: show surrounding lines for each match
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional


def tool_info() -> dict:
    return {
        "name": "str_replace_editor",
        "description": (
            "Advanced string replacement editor with preview and multi-replace support. "
            "Commands: preview (show changes without applying), replace (apply changes), "
            "regex_replace (pattern-based replacement), multi_replace (replace all occurrences). "
            "Provides context around matches for better visibility."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["preview", "replace", "regex_replace", "multi_replace"],
                    "description": "The command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file.",
                },
                "old_str": {
                    "type": "string",
                    "description": "String or pattern to find.",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show around matches (default: 3).",
                    "default": 3,
                },
                "max_replacements": {
                    "type": "integer",
                    "description": "Maximum number of replacements for multi_replace (default: 10).",
                    "default": 10,
                },
            },
            "required": ["command", "path", "old_str"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict editor operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def _get_match_context(content: str, match_start: int, match_end: int, context_lines: int = 3) -> Tuple[int, int, str]:
    """Get context lines around a match.
    
    Returns:
        (start_line_num, end_line_num, context_str)
    """
    lines = content.split('\n')
    
    # Find which line the match starts on
    char_count = 0
    start_line = 0
    for i, line in enumerate(lines):
        if char_count + len(line) + 1 > match_start:  # +1 for newline
            start_line = i
            break
        char_count += len(line) + 1
    
    # Find which line the match ends on
    char_count = 0
    end_line = 0
    for i, line in enumerate(lines):
        if char_count + len(line) + 1 >= match_end:
            end_line = i
            break
        char_count += len(line) + 1
    
    # Calculate context range
    context_start = max(0, start_line - context_lines)
    context_end = min(len(lines), end_line + context_lines + 1)
    
    # Build context string with line numbers
    context_parts = []
    for i in range(context_start, context_end):
        prefix = ">>> " if context_start <= i <= end_line else "    "
        context_parts.append(f"{prefix}{i + 1:4d}: {lines[i]}")
    
    return context_start + 1, context_end, '\n'.join(context_parts)


def _find_all_matches(content: str, old_str: str) -> List[Tuple[int, int]]:
    """Find all occurrences of old_str in content.
    
    Returns:
        List of (start, end) tuples for each match.
    """
    matches = []
    start = 0
    while True:
        idx = content.find(old_str, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(old_str)))
        start = idx + 1
    return matches


def _preview_replace(content: str, old_str: str, new_str: str, context_lines: int = 3) -> str:
    """Generate a preview of the replacement without applying it."""
    matches = _find_all_matches(content, old_str)
    
    if not matches:
        return f"No matches found for the search string.\nSearched for:\n{old_str[:200]}{'...' if len(old_str) > 200 else ''}"
    
    result = [f"Found {len(matches)} occurrence(s) of the search string:\n"]
    
    for i, (match_start, match_end) in enumerate(matches, 1):
        start_line, end_line, context = _get_match_context(content, match_start, match_end, context_lines)
        
        # Show the actual replacement
        result.append(f"\n--- Match #{i} (lines {start_line}-{end_line}) ---")
        result.append(context)
        result.append(f"\nWould replace with:\n{new_str[:200]}{'...' if len(new_str) > 200 else ''}")
        result.append("-" * 50)
    
    return '\n'.join(result)


def _do_replace(content: str, old_str: str, new_str: str) -> Tuple[str, int]:
    """Perform the replacement and return new content + count."""
    count = content.count(old_str)
    if count == 0:
        return content, 0
    if count > 1:
        # Only replace first occurrence
        idx = content.find(old_str)
        new_content = content[:idx] + new_str + content[idx + len(old_str):]
        return new_content, 1
    new_content = content.replace(old_str, new_str)
    return new_content, count


def _do_multi_replace(content: str, old_str: str, new_str: str, max_replacements: int = 10) -> Tuple[str, int]:
    """Replace all occurrences up to max_replacements."""
    count = 0
    new_content = content
    for _ in range(max_replacements):
        if old_str not in new_content:
            break
        idx = new_content.find(old_str)
        new_content = new_content[:idx] + new_str + new_content[idx + len(old_str):]
        count += 1
    return new_content, count


def _do_regex_replace(content: str, pattern: str, new_str: str, max_replacements: int = 10) -> Tuple[str, int]:
    """Perform regex-based replacement."""
    try:
        new_content, count = re.subn(pattern, new_str, content, count=max_replacements)
        return new_content, count
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")


def tool_function(
    command: str,
    path: str,
    old_str: str,
    new_str: str = "",
    context_lines: int = 3,
    max_replacements: int = 10,
) -> str:
    """Execute an advanced string replacement command."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if not old_str:
            return "Error: old_str cannot be empty."
        
        content = p.read_text()
        
        if command == "preview":
            return _preview_replace(content, old_str, new_str, context_lines)
        
        elif command == "replace":
            matches = _find_all_matches(content, old_str)
            if not matches:
                return f"Error: old_str not found in {p}"
            if len(matches) > 1:
                return f"Error: old_str appears {len(matches)} times. Use 'multi_replace' to replace all, or make old_str more specific."
            
            new_content, count = _do_replace(content, old_str, new_str)
            p.write_text(new_content)
            
            # Show context of the change
            idx = content.find(old_str)
            start_line, end_line, context = _get_match_context(content, idx, idx + len(old_str), context_lines)
            return f"File {p} edited successfully. Replaced 1 occurrence at lines {start_line}-{end_line}.\n\nContext:\n{context}"
        
        elif command == "multi_replace":
            matches = _find_all_matches(content, old_str)
            if not matches:
                return f"Error: old_str not found in {p}"
            
            new_content, count = _do_multi_replace(content, old_str, new_str, max_replacements)
            p.write_text(new_content)
            
            return f"File {p} edited successfully. Replaced {count} occurrence(s) (max allowed: {max_replacements})."
        
        elif command == "regex_replace":
            try:
                # Preview first
                preview_count = len(re.findall(old_str, content))
                if preview_count == 0:
                    return f"No matches found for regex pattern: {old_str}"
                
                new_content, count = _do_regex_replace(content, old_str, new_str, max_replacements)
                p.write_text(new_content)
                
                return f"File {p} edited successfully. Replaced {count} occurrence(s) using regex pattern."
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"
        
        else:
            return f"Error: unknown command {command}"
            
    except Exception as e:
        return f"Error: {e}"
