"""
Code search tool for analyzing and searching through the codebase.

Provides functionality to:
- Search for patterns in code files
- Find function/class definitions
- Analyze code structure
- Extract code snippets with context
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any


def _get_python_files(repo_path: str) -> list[str]:
    """Get all Python files in the repository, excluding __pycache__ and .pyc files."""
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for file in files:
            if file.endswith(".py") and not file.endswith(".pyc"):
                python_files.append(os.path.join(root, file))
    return python_files


def _read_file_with_context(file_path: str, line_num: int, context: int = 5) -> str:
    """Read a file and extract lines around a specific line number."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)
        
        result = []
        for i in range(start, end):
            marker = ">>> " if i == line_num - 1 else "    "
            result.append(f"{marker}{i+1:4d}: {lines[i]}")
        
        return "".join(result)
    except Exception as e:
        return f"Error reading file: {e}"


def search_code(
    repo_path: str,
    pattern: str,
    file_pattern: str = "*.py",
    context_lines: int = 3,
) -> str:
    """Search for a pattern in code files.

    Args:
        repo_path: Path to the repository to search
        pattern: Regular expression pattern to search for
        file_pattern: Glob pattern for files to search (default: "*.py")
        context_lines: Number of context lines to include (default: 3)

    Returns:
        Search results with file paths, line numbers, and context
    """
    if not os.path.isdir(repo_path):
        return f"Error: {repo_path} is not a valid directory"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results = []
    files_searched = 0
    matches_found = 0

    for root, dirs, files in os.walk(repo_path):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for file in files:
            if not file.endswith(".py") or file.endswith(".pyc"):
                continue
            
            file_path = os.path.join(root, file)
            files_searched += 1
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")
                
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches_found += 1
                        # Get context
                        start = max(0, i - context_lines - 1)
                        end = min(len(lines), i + context_lines)
                        
                        context_lines_str = []
                        for j in range(start, end):
                            marker = ">>> " if j == i - 1 else "    "
                            context_lines_str.append(f"{marker}{j+1:4d}: {lines[j]}")
                        
                        rel_path = os.path.relpath(file_path, repo_path)
                        results.append(f"\n=== {rel_path}:{i} ===\n" + "\n".join(context_lines_str))
                        
                        # Limit results per file to avoid overwhelming output
                        if len([r for r in results if rel_path in r]) >= 10:
                            break
                            
            except Exception as e:
                continue

    if not results:
        return f"No matches found for pattern '{pattern}' in {files_searched} files searched."

    header = f"Found {matches_found} matches in {files_searched} files:\n"
    return header + "\n".join(results[:50])  # Limit total results


def find_function_definitions(
    repo_path: str,
    function_name: str | None = None,
    class_name: str | None = None,
) -> str:
    """Find function or class definitions in the codebase.

    Args:
        repo_path: Path to the repository to search
        function_name: Name of function to find (optional)
        class_name: Name of class to find (optional)

    Returns:
        List of definitions with file paths and line numbers
    """
    if not os.path.isdir(repo_path):
        return f"Error: {repo_path} is not a valid directory"

    results = []
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for file in files:
            if not file.endswith(".py"):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if function_name is None or node.name == function_name:
                            # Get function signature
                            args_str = ast.unparse(node.args) if hasattr(ast, "unparse") else "..."
                            docstring = ast.get_docstring(node)
                            doc_preview = f'"""{docstring[:100]}..."""' if docstring else ""
                            
                            results.append(
                                f"\nFunction: {node.name}({args_str})\n"
                                f"  File: {rel_path}:{node.lineno}\n"
                                f"  Docstring: {doc_preview}"
                            )
                    
                    elif isinstance(node, ast.ClassDef):
                        if class_name is None or node.name == class_name:
                            docstring = ast.get_docstring(node)
                            doc_preview = f'"""{docstring[:100]}..."""' if docstring else ""
                            
                            # Get methods
                            methods = [
                                n.name for n in node.body 
                                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
                            ]
                            
                            results.append(
                                f"\nClass: {node.name}\n"
                                f"  File: {rel_path}:{node.lineno}\n"
                                f"  Docstring: {doc_preview}\n"
                                f"  Public methods: {', '.join(methods[:5])}"
                                f"{'...' if len(methods) > 5 else ''}"
                            )
                            
            except SyntaxError:
                continue
            except Exception as e:
                continue

    if not results:
        search_target = function_name or class_name or "any functions or classes"
        return f"No definitions found for {search_target}"

    return f"Found {len(results)} definitions:\n" + "\n".join(results[:30])


def analyze_code_structure(repo_path: str) -> str:
    """Analyze the overall structure of a Python codebase.

    Args:
        repo_path: Path to the repository to analyze

    Returns:
        Summary of code structure including file count, function count, class count
    """
    if not os.path.isdir(repo_path):
        return f"Error: {repo_path} is not a valid directory"

    stats = {
        "files": 0,
        "functions": 0,
        "classes": 0,
        "imports": 0,
        "lines_of_code": 0,
    }
    
    file_list = []
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        
        for file in files:
            if not file.endswith(".py"):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                lines = content.split("\n")
                stats["lines_of_code"] += len([l for l in lines if l.strip()])
                stats["files"] += 1
                
                tree = ast.parse(content)
                
                file_info = {
                    "path": rel_path,
                    "functions": [],
                    "classes": [],
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        stats["functions"] += 1
                        if node.lineno < 20:  # Only show top-level functions
                            file_info["functions"].append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        stats["classes"] += 1
                        file_info["classes"].append(node.name)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        stats["imports"] += 1
                
                if file_info["functions"] or file_info["classes"]:
                    file_list.append(file_info)
                    
            except Exception:
                continue

    summary = f"""Codebase Analysis Summary:
========================
Total Files: {stats["files"]}
Total Functions: {stats["functions"]}
Total Classes: {stats["classes"]}
Total Imports: {stats["imports"]}
Lines of Code: {stats["lines_of_code"]}

Key Files:
"""
    
    for f in sorted(file_list, key=lambda x: len(x["functions"]) + len(x["classes"]), reverse=True)[:10]:
        summary += f"\n  {f['path']}"
        if f["classes"]:
            summary += f"\n    Classes: {', '.join(f['classes'])}"
        if f["functions"]:
            summary += f"\n    Functions: {', '.join(f['functions'][:5])}"
            if len(f["functions"]) > 5:
                summary += f"... ({len(f['functions'])} total)"
    
    return summary


# Tool definitions for the agent framework
tool_info = {
    "search_code": {
        "name": "search_code",
        "description": "Search for patterns in code files using regular expressions. Returns file paths, line numbers, and context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to search",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to search (default: '*.py')",
                    "default": "*.py",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to include (default: 3)",
                    "default": 3,
                },
            },
            "required": ["repo_path", "pattern"],
        },
    },
    "find_function_definitions": {
        "name": "find_function_definitions",
        "description": "Find function or class definitions in the codebase. Can search for specific names or list all definitions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to search",
                },
                "function_name": {
                    "type": "string",
                    "description": "Name of function to find (optional, lists all if not provided)",
                },
                "class_name": {
                    "type": "string",
                    "description": "Name of class to find (optional, lists all if not provided)",
                },
            },
            "required": ["repo_path"],
        },
    },
    "analyze_code_structure": {
        "name": "analyze_code_structure",
        "description": "Analyze the overall structure of a Python codebase. Returns statistics and key file summaries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to analyze",
                },
            },
            "required": ["repo_path"],
        },
    },
}


def get_tools():
    """Return tool definitions for registration."""
    return [
        {"info": tool_info["search_code"], "function": search_code},
        {"info": tool_info["find_function_definitions"], "function": find_function_definitions},
        {"info": tool_info["analyze_code_structure"], "function": analyze_code_structure},
    ]
