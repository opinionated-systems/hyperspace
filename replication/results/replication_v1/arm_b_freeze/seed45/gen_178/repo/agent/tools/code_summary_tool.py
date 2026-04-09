"""
Code summary tool: extract high-level structure and documentation from Python files.

Provides a quick overview of Python files including:
- Module docstring
- Classes and their methods
- Functions with their signatures
- Imports
- Key decorators

This helps the meta agent quickly understand code structure without reading entire files.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_summary",
        "description": (
            "Extract high-level structure from Python files. "
            "Shows classes, functions, imports, and docstrings. "
            "Useful for quickly understanding code structure without reading entire files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file or directory.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for nested structures (default: 2).",
                },
                "include_docstrings": {
                    "type": "boolean",
                    "description": "Whether to include docstrings in summary (default: true).",
                },
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict code summary operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_list(items: list[str], max_items: int = 20) -> list[str]:
    """Truncate list to max_items."""
    if len(items) > max_items:
        return items[:max_items] + [f"... ({len(items) - max_items} more items)"]
    return items


def tool_function(
    path: str,
    max_depth: int = 2,
    include_docstrings: bool = True,
) -> str:
    """Generate a summary of Python code structure."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if p.is_dir():
            return _summarize_directory(p, max_depth, include_docstrings)
        elif p.suffix == ".py":
            return _summarize_file(p, max_depth, include_docstrings)
        else:
            return f"Error: {p} is not a Python file or directory."
            
    except Exception as e:
        return f"Error: {e}"


def _summarize_directory(path: Path, max_depth: int, include_docstrings: bool) -> str:
    """Summarize all Python files in a directory."""
    py_files = list(path.rglob("*.py"))
    
    # Exclude common non-source directories
    exclude_patterns = [
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".pytest_cache", ".mypy_cache", ".tox", "build", "dist"
    ]
    py_files = [
        f for f in py_files 
        if not any(pattern in str(f) for pattern in exclude_patterns)
    ]
    
    if not py_files:
        return f"No Python files found in {path}"
    
    # Sort and limit
    py_files.sort()
    total_files = len(py_files)
    
    results = [f"Code Summary for {path}:", "=" * 60]
    results.append(f"Total Python files: {total_files}\n")
    
    # Show first 10 files with summaries
    for file_path in py_files[:10]:
        file_summary = _summarize_file_internal(file_path, max_depth, include_docstrings)
        results.append(f"\n{file_summary}")
    
    if len(py_files) > 10:
        results.append(f"\n... and {len(py_files) - 10} more files")
    
    return "\n".join(results)


def _summarize_file(path: Path, max_depth: int, include_docstrings: bool) -> str:
    """Summarize a single Python file."""
    return _summarize_file_internal(path, max_depth, include_docstrings)


def _summarize_file_internal(path: Path, max_depth: int, include_docstrings: bool) -> str:
    """Internal function to generate file summary."""
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading {path}: {e}"
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Syntax error in {path} at line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"Parse error in {path}: {e}"
    
    lines = [f"File: {path}", "-" * 40]
    
    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc and include_docstrings:
        doc_preview = module_doc[:200] + "..." if len(module_doc) > 200 else module_doc
        lines.append(f"\nModule docstring: {doc_preview}")
    
    # Imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append(f"import {name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(f"from {module} import {', '.join(names)}")
    
    if imports:
        lines.append(f"\nImports ({len(imports)}):")
        for imp in _truncate_list(imports, 10):
            lines.append(f"  {imp}")
    
    # Classes
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if classes:
        lines.append(f"\nClasses ({len(classes)}):")
        for cls in classes:
            # Get base classes
            bases = []
            for base in cls.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{base.value.id}.{base.attr}")
            
            base_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"  class {cls.name}{base_str}:")
            
            # Methods
            methods = [
                n for n in cls.body 
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            
            if methods and max_depth > 1:
                for method in methods[:5]:  # Show first 5 methods
                    sig = _get_function_signature(method)
                    lines.append(f"    def {sig}")
                if len(methods) > 5:
                    lines.append(f"    ... and {len(methods) - 5} more methods")
    
    # Top-level functions
    top_functions = [
        n for n in tree.body 
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    
    if top_functions:
        lines.append(f"\nFunctions ({len(top_functions)}):")
        for func in top_functions[:10]:  # Show first 10 functions
            sig = _get_function_signature(func)
            lines.append(f"  def {sig}")
        if len(top_functions) > 10:
            lines.append(f"  ... and {len(top_functions) - 10} more functions")
    
    # Summary stats
    lines.append(f"\nSummary: {len(classes)} classes, {len(top_functions)} functions, {len(imports)} imports")
    
    return "\n".join(lines)


def _get_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract function signature as a string."""
    name = node.name
    
    # Get arguments
    args = node.args
    arg_strings = []
    
    # Positional args
    for i, arg in enumerate(args.args):
        arg_name = arg.arg
        # Check for default
        default_idx = len(args.args) - len(args.defaults) + i
        if default_idx >= len(args.args) - len(args.defaults) and args.defaults:
            default_idx = i - (len(args.args) - len(args.defaults))
            if default_idx >= 0:
                default = args.defaults[default_idx]
                default_str = _get_default_str(default)
                arg_strings.append(f"{arg_name}={default_str}")
            else:
                arg_strings.append(arg_name)
        else:
            arg_strings.append(arg_name)
    
    # *args
    if args.vararg:
        arg_strings.append(f"*{args.vararg.arg}")
    
    # Keyword-only args
    for arg in args.kwonlyargs:
        arg_strings.append(arg.arg)
    
    # **kwargs
    if args.kwarg:
        arg_strings.append(f"**{args.kwarg.arg}")
    
    # Check for async
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    
    return f"{prefix}{name}({', '.join(arg_strings)})"


def _get_default_str(node: ast.AST) -> str:
    """Get string representation of a default value."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.NameConstant):
        return repr(node.value)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Str):
        return repr(node.s)
    elif isinstance(node, ast.Num):
        return repr(node.n)
    elif isinstance(node, ast.List):
        return "[]"
    elif isinstance(node, ast.Dict):
        return "{}"
    elif isinstance(node, ast.Set):
        return "set()"
    else:
        return "..."
