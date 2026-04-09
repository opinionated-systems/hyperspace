"""
Search tool for searching text within files.

This tool provides functionality to search for patterns in files,
similar to grep functionality.
"""

import os
import re
from typing import List, Dict, Any, Optional


def search_in_file(
    file_path: str,
    pattern: str,
    case_sensitive: bool = True,
    max_results: int = 100
) -> Dict[str, Any]:
    """
    Search for a pattern within a specific file.
    
    Args:
        file_path: Path to the file to search
        pattern: The regex pattern to search for
        case_sensitive: Whether the search should be case sensitive
        max_results: Maximum number of matches to return
        
    Returns:
        Dictionary with search results
    """
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "matches": []
        }
    
    if not os.path.isfile(file_path):
        return {
            "success": False,
            "error": f"Path is not a file: {file_path}",
            "matches": []
        }
    
    flags = 0 if case_sensitive else re.IGNORECASE
    
    try:
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "matches": []
        }
    
    matches = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if len(matches) >= max_results:
                    break
                    
                for match in compiled_pattern.finditer(line):
                    matches.append({
                        "line_number": line_num,
                        "line_content": line.rstrip('\n\r'),
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "matched_text": match.group(0)
                    })
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {e}",
            "matches": matches
        }
    
    return {
        "success": True,
        "file_path": file_path,
        "pattern": pattern,
        "total_matches": len(matches),
        "matches": matches
    }


def search_in_directory(
    directory: str,
    pattern: str,
    file_extensions: Optional[List[str]] = None,
    case_sensitive: bool = True,
    max_results: int = 100,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Search for a pattern across multiple files in a directory.
    
    Args:
        directory: Path to the directory to search
        pattern: The regex pattern to search for
        file_extensions: List of file extensions to include (e.g., ['.py', '.txt'])
        case_sensitive: Whether the search should be case sensitive
        max_results: Maximum number of matches to return
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        Dictionary with search results
    """
    if not os.path.exists(directory):
        return {
            "success": False,
            "error": f"Directory not found: {directory}",
            "matches": []
        }
    
    if not os.path.isdir(directory):
        return {
            "success": False,
            "error": f"Path is not a directory: {directory}",
            "matches": []
        }
    
    flags = 0 if case_sensitive else re.IGNORECASE
    
    try:
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "matches": []
        }
    
    all_matches = []
    files_searched = 0
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if file_extensions and not any(filename.endswith(ext) for ext in file_extensions):
                    continue
                    
                file_path = os.path.join(root, filename)
                files_searched += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if len(all_matches) >= max_results:
                                break
                                
                            for match in compiled_pattern.finditer(line):
                                all_matches.append({
                                    "file_path": file_path,
                                    "line_number": line_num,
                                    "line_content": line.rstrip('\n\r'),
                                    "match_start": match.start(),
                                    "match_end": match.end(),
                                    "matched_text": match.group(0)
                                })
                except Exception:
                    # Skip files that can't be read
                    pass
                    
                if len(all_matches) >= max_results:
                    break
            
            if len(all_matches) >= max_results:
                break
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue
                
            if file_extensions and not any(filename.endswith(ext) for ext in file_extensions):
                continue
            
            files_searched += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if len(all_matches) >= max_results:
                            break
                            
                        for match in compiled_pattern.finditer(line):
                            all_matches.append({
                                "file_path": file_path,
                                "line_number": line_num,
                                "line_content": line.rstrip('\n\r'),
                                "match_start": match.start(),
                                "match_end": match.end(),
                                "matched_text": match.group(0)
                            })
            except Exception:
                # Skip files that can't be read
                pass
                
            if len(all_matches) >= max_results:
                break
    
    return {
        "success": True,
        "directory": directory,
        "pattern": pattern,
        "files_searched": files_searched,
        "total_matches": len(all_matches),
        "matches": all_matches
    }


# Tool schema for LLM tool calling
SEARCH_IN_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_in_file",
        "description": "Search for a regex pattern within a specific file and return matching lines with context",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to search"
                },
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive (default: True)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 100)"
                }
            },
            "required": ["file_path", "pattern"]
        }
    }
}

SEARCH_IN_DIRECTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_in_directory",
        "description": "Search for a regex pattern across multiple files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Absolute path to the directory to search"
                },
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of file extensions to include (e.g., ['.py', '.txt'])"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive (default: True)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 100)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories (default: True)"
                }
            },
            "required": ["directory", "pattern"]
        }
    }
}


# Tool interface for registry
def tool_info():
    """Return tool information for registry."""
    return {
        "name": "search",
        "description": "Search for text patterns in files and directories",
        "functions": [
            SEARCH_IN_FILE_SCHEMA,
            SEARCH_IN_DIRECTORY_SCHEMA,
        ]
    }


def tool_function(name: str, **kwargs):
    """Execute a search tool function by name."""
    if name == "search_in_file":
        return search_in_file(**kwargs)
    elif name == "search_in_directory":
        return search_in_directory(**kwargs)
    else:
        raise ValueError(f"Unknown search tool function: {name}")
