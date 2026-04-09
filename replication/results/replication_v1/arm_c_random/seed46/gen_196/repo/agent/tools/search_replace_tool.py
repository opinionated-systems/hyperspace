"""
Search and replace tool: combines search functionality with replace operations.

This tool provides a convenient way to search for content and replace it
in a single operation, with preview capabilities.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent.tools import editor_tool


def tool_info() -> dict:
    return {
        "name": "search_replace",
        "description": (
            "Search for text in files and replace matches. "
            "Supports single file or directory-wide search and replace. "
            "Can preview changes before applying them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory to search in.",
                },
                "search_term": {
                    "type": "string",
                    "description": "Text to search for.",
                },
                "replacement": {
                    "type": "string",
                    "description": "Text to replace matches with. If not provided, only search is performed.",
                },
                "preview": {
                    "type": "boolean",
                    "description": "If true, show what would be changed without applying. Default: false.",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum number of replacements to make. Default: 100.",
                },
            },
            "required": ["path", "search_term"],
        },
    }


def tool_function(
    path: str,
    search_term: str,
    replacement: str | None = None,
    preview: bool = False,
    max_matches: int = 100,
) -> str:
    """Search for text and optionally replace matches."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {p} does not exist."

        # If no replacement provided, just do a search
        if replacement is None:
            return editor_tool._search(p, search_term)

        matches_found = 0
        replacements_made = 0
        results = []

        if p.is_dir():
            # Search and replace in directory
            for root, _, files in os.walk(p):
                # Skip hidden directories
                if "/." in root:
                    continue
                for file in files:
                    if file.startswith("."):
                        continue
                    file_path = Path(root) / file
                    try:
                        content = file_path.read_text(errors="ignore")
                        if search_term in content:
                            count = content.count(search_term)
                            matches_found += count
                            
                            if not preview and replacements_made < max_matches:
                                # Perform replacement
                                new_content = content.replace(search_term, replacement)
                                file_path.write_text(new_content)
                                replacements_made += min(count, max_matches - replacements_made)
                                results.append(f"  {file_path}: replaced {count} occurrence(s)")
                            else:
                                results.append(f"  {file_path}: would replace {count} occurrence(s)")
                            
                            if replacements_made >= max_matches:
                                break
                    except Exception as e:
                        results.append(f"  {file_path}: error - {e}")
                        continue
                if replacements_made >= max_matches:
                    break
        else:
            # Single file
            content = p.read_text()
            count = content.count(search_term)
            matches_found = count
            
            if count > 0:
                if not preview:
                    new_content = content.replace(search_term, replacement)
                    p.write_text(new_content)
                    replacements_made = count
                    results.append(f"  {p}: replaced {count} occurrence(s)")
                else:
                    results.append(f"  {p}: would replace {count} occurrence(s)")
            else:
                return f"No matches found for '{search_term}' in {p}"

        # Build result message
        mode_str = "Would replace" if preview else "Replaced"
        header = f"{mode_str} {matches_found} match(es) for '{search_term}'"
        if preview:
            header += " (preview mode - no changes made)"
        
        if results:
            return header + ":\n" + "\n".join(results)
        else:
            return f"No matches found for '{search_term}' in {p}"

    except Exception as e:
        return f"Error: {e}"
