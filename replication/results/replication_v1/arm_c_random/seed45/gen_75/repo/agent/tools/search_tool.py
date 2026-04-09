"""
Search tool: find text patterns in files.

Provides grep-like functionality to search for patterns across the codebase.
Useful for finding code locations before editing.

Enhanced with Python AST support for intelligent code searching.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from agent.config import DEFAULT_AGENT_CONFIG


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files. "
            "Uses grep-like syntax for finding code. "
            "Returns matching lines with file paths and line numbers. "
            "For Python files, supports special patterns: "
            "'class:Name' to find class definitions, "
            "'def:name' to find function definitions, "
            "'func:name' to find function calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported). For Python AST search, use 'class:Name', 'def:name', or 'func:name'.",
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
                    "description": "Whether search is case sensitive (default: true).",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate_results(results: str, max_lines: int = 100) -> str:
    """Truncate results to max_lines."""
    lines = results.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return results


class _PythonASTVisitor(ast.NodeVisitor):
    """AST visitor for finding Python constructs."""
    
    def __init__(self, target_name: str, search_type: str) -> None:
        self.target_name = target_name
        self.search_type = search_type  # 'class', 'def', or 'func'
        self.results: list[tuple[str, int, str]] = []  # (file_path, line, context)
        self.file_path: str = ""
        self.source_lines: list[str] = []
    
    def set_file(self, file_path: str, source: str) -> None:
        """Set the current file being analyzed."""
        self.file_path = file_path
        self.source_lines = source.split("\n")
        self.results = []
    
    def _get_context(self, node: ast.AST, lines_before: int = 2, lines_after: int = 2) -> str:
        """Get source context around a node."""
        start_line = getattr(node, 'lineno', 1) - 1
        end_line = getattr(node, 'end_lineno', start_line + 1)
        
        context_start = max(0, start_line - lines_before)
        context_end = min(len(self.source_lines), end_line + lines_after)
        
        lines = []
        for i in range(context_start, context_end):
            prefix = ">>> " if context_start <= i < end_line else "    "
            lines.append(f"{prefix}{i + 1:4d}: {self.source_lines[i]}")
        return "\n".join(lines)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        if self.search_type == 'class' and node.name == self.target_name:
            context = self._get_context(node)
            self.results.append((self.file_path, node.lineno, context))
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        if self.search_type == 'def' and node.name == self.target_name:
            context = self._get_context(node)
            self.results.append((self.file_path, node.lineno, context))
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        if self.search_type == 'def' and node.name == self.target_name:
            context = self._get_context(node)
            self.results.append((self.file_path, node.lineno, context))
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        if self.search_type == 'func':
            func_name = self._get_call_name(node.func)
            if func_name == self.target_name:
                context = self._get_context(node)
                self.results.append((self.file_path, node.lineno, context))
        self.generic_visit(node)
    
    def _get_call_name(self, node: ast.AST) -> str:
        """Extract the name from a call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_call_name(node.value)}.{node.attr}"
        return ""


def _search_python_ast(
    pattern: str,
    path: Path,
    file_extension: str | None = None,
) -> str | None:
    """Search Python files using AST for class/def/func patterns.
    
    Pattern formats:
    - 'class:Name' - find class definitions
    - 'def:name' - find function definitions  
    - 'func:name' - find function calls
    
    Returns formatted results or None if not an AST pattern.
    """
    # Check if pattern is an AST search pattern
    ast_prefixes = ('class:', 'def:', 'func:')
    if not any(pattern.startswith(prefix) for prefix in ast_prefixes):
        return None
    
    # Parse the pattern
    if ':' not in pattern:
        return None
    
    search_type, target_name = pattern.split(':', 1)
    if not target_name:
        return "Error: Empty target name in AST pattern"
    
    visitor = _PythonASTVisitor(target_name, search_type)
    all_results: list[tuple[str, int, str]] = []
    
    # Find Python files
    if path.is_file():
        if path.suffix == '.py':
            files = [path]
        else:
            return f"Error: AST search only works on Python files, got {path.suffix}"
    else:
        files = list(path.rglob('*.py'))
        if file_extension and file_extension != '.py':
            return f"Error: AST search only works on Python files, got {file_extension}"
    
    # Analyze each file
    for file_path in files:
        try:
            source = file_path.read_text(encoding='utf-8', errors='ignore')
            try:
                tree = ast.parse(source)
                visitor.set_file(str(file_path), source)
                visitor.visit(tree)
                # Collect results from this file
                all_results.extend(visitor.results)
                # Clear results for next file but keep the visitor
                visitor.results = []
            except SyntaxError:
                # Skip files with syntax errors
                continue
        except Exception:
            continue
    
    # Format results
    if not all_results:
        return f"No {search_type} '{target_name}' found in {path}"
    
    lines = [f"Found {len(all_results)} match(es) for {search_type} '{target_name}':"]
    for file_path, line_num, context in all_results:
        lines.append(f"\n{file_path}:{line_num}")
        lines.append(context)
    
    return _truncate_results("\n".join(lines), max_lines=50)


def tool_function(
    pattern: str,
    path: str,
    file_extension: str | None = None,
    case_sensitive: bool = True,
) -> str:
    """Search for pattern in files.
    
    Supports two search modes:
    1. Text/regex search (default): Uses grep for pattern matching
    2. Python AST search: For Python-specific constructs using patterns like:
       - 'class:Name' - find class definitions
       - 'def:name' - find function definitions
       - 'func:name' - find function calls
    
    Args:
        pattern: Regex pattern to search for, or AST pattern (class:/def:/func:)
        path: Directory or file to search in
        file_extension: Optional extension filter (e.g., '.py')
        case_sensitive: Whether search is case sensitive (text search only)
    
    Returns:
        Matching lines with file:line format, or AST search results with context
    """
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {path} does not exist."

        # Try Python AST search first (for class:/def:/func: patterns)
        ast_result = _search_python_ast(pattern, p, file_extension)
        if ast_result is not None:
            return ast_result

        # Fall back to grep-based search
        cmd = ["grep", "-rn" if case_sensitive else "-rni"]
        
        # Add file extension filter if specified
        if file_extension:
            cmd.extend(["--include", f"*{file_extension}"])
        
        # Add pattern and path
        cmd.extend([pattern, str(p)])

        # Run search
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Found matches
            return _truncate_results(result.stdout.strip())
        elif result.returncode == 1:
            # No matches found
            return f"No matches found for pattern '{pattern}' in {path}"
        else:
            # Error
            return f"Error searching: {result.stderr}"

    except Exception as e:
        return f"Error: {e}"
