"""
Context tool: provides intelligent code context and summaries.

Helps the meta-agent understand code relationships by providing:
- Function/class summaries
- Import/dependency analysis
- Code structure overview
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "context",
        "description": (
            "Provides intelligent code context and summaries. "
            "Analyzes Python files to show function/class definitions, "
            "imports, and code structure. Useful for understanding "
            "code relationships before making modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Python file to analyze.",
                },
                "command": {
                    "type": "string",
                    "enum": ["summary", "imports", "functions", "classes"],
                    "description": "Type of context to provide.",
                },
            },
            "required": ["path", "command"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for context analysis."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def tool_function(path: str, command: str) -> str:
    """Provide code context and analysis.
    
    Args:
        path: Absolute path to the Python file
        command: Type of analysis (summary, imports, functions, classes)
    
    Returns:
        Formatted context information
    """
    # Validate inputs
    if not path or not isinstance(path, str):
        return "Error: path must be a non-empty string"
    
    valid_commands = ["summary", "imports", "functions", "classes"]
    if command not in valid_commands:
        return f"Error: Invalid command '{command}'. Use: {', '.join(valid_commands)}"
    
    # Check path is within allowed root
    if not _is_within_allowed(path):
        return f"Error: Path '{path}' is outside allowed root."
    
    p = Path(path)
    
    if not p.exists():
        return f"Error: File '{path}' does not exist."
    
    if not p.is_file():
        return f"Error: '{path}' is not a file."
    
    if not path.endswith('.py'):
        return f"Error: Context tool only supports Python files (.py)."
    
    try:
        content = p.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file: {type(e).__name__}: {e}"
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error parsing Python syntax: {e}"
    except Exception as e:
        return f"Error analyzing file: {type(e).__name__}: {e}"
    
    if command == "summary":
        return _get_summary(tree, path)
    elif command == "imports":
        return _get_imports(tree, path)
    elif command == "functions":
        return _get_functions(tree, path)
    elif command == "classes":
        return _get_classes(tree, path)
    
    return f"Error: Unknown command '{command}'"


def _get_summary(tree: ast.AST, path: str) -> str:
    """Get a summary of the Python file."""
    lines = [f"Code Summary for: {path}", "=" * 50, ""]
    
    # Count various elements
    imports = []
    functions = []
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
    
    lines.append(f"Total Imports: {len(imports)}")
    lines.append(f"Total Functions: {len(functions)}")
    lines.append(f"Total Classes: {len(classes)}")
    lines.append("")
    
    # List top-level functions
    if functions:
        lines.append("Top-level Functions:")
        for func in functions:
            if func.col_offset == 0:  # Top-level
                args = [arg.arg for arg in func.args.args]
                lines.append(f"  - {func.name}({', '.join(args)})")
        lines.append("")
    
    # List classes with methods
    if classes:
        lines.append("Classes:")
        for cls in classes:
            methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
            lines.append(f"  - {cls.name}")
            if methods:
                lines.append(f"    Methods: {', '.join(methods)}")
        lines.append("")
    
    return "\n".join(lines)


def _get_imports(tree: ast.AST, path: str) -> str:
    """Get all imports from the Python file."""
    lines = [f"Imports in: {path}", "=" * 50, ""]
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            if module:
                imports.append(f"from {module} import {', '.join(names)}")
            else:
                imports.append(f"import {', '.join(names)}")
    
    if imports:
        lines.append("Import Statements:")
        for imp in sorted(set(imports)):
            lines.append(f"  {imp}")
    else:
        lines.append("No imports found.")
    
    return "\n".join(lines)


def _get_functions(tree: ast.AST, path: str) -> str:
    """Get detailed function information."""
    lines = [f"Functions in: {path}", "=" * 50, ""]
    
    functions_found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function signature
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)
            
            # Get decorators
            decorators = [ast.unparse(d) for d in node.decorator_list]
            
            # Get docstring
            docstring = ast.get_docstring(node)
            
            functions_found.append({
                'name': node.name,
                'args': args,
                'line': node.lineno,
                'decorators': decorators,
                'docstring': docstring,
                'is_method': node.col_offset > 0,
            })
    
    if functions_found:
        for func in sorted(functions_found, key=lambda x: x['line']):
            lines.append(f"Function: {func['name']} (line {func['line']})")
            if func['decorators']:
                lines.append(f"  Decorators: {', '.join(func['decorators'])}")
            lines.append(f"  Args: {', '.join(func['args'])}")
            if func['docstring']:
                # Truncate long docstrings
                doc = func['docstring'][:100] + "..." if len(func['docstring']) > 100 else func['docstring']
                lines.append(f"  Docstring: {doc}")
            if func['is_method']:
                lines.append("  [Method of a class]")
            lines.append("")
    else:
        lines.append("No functions found.")
    
    return "\n".join(lines)


def _get_classes(tree: ast.AST, path: str) -> str:
    """Get detailed class information."""
    lines = [f"Classes in: {path}", "=" * 50, ""]
    
    classes_found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get base classes
            bases = [ast.unparse(base) for base in node.bases]
            
            # Get methods
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
            
            # Get docstring
            docstring = ast.get_docstring(node)
            
            classes_found.append({
                'name': node.name,
                'line': node.lineno,
                'bases': bases,
                'methods': methods,
                'docstring': docstring,
            })
    
    if classes_found:
        for cls in sorted(classes_found, key=lambda x: x['line']):
            lines.append(f"Class: {cls['name']} (line {cls['line']})")
            if cls['bases']:
                lines.append(f"  Inherits from: {', '.join(cls['bases'])}")
            if cls['methods']:
                lines.append(f"  Methods: {', '.join(cls['methods'])}")
            if cls['docstring']:
                doc = cls['docstring'][:100] + "..." if len(cls['docstring']) > 100 else cls['docstring']
                lines.append(f"  Docstring: {doc}")
            lines.append("")
    else:
        lines.append("No classes found.")
    
    return "\n".join(lines)
