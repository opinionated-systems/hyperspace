"""
Code analysis tool: analyze Python code structure and extract information.

Provides functionality to parse Python files and extract:
- Function/class definitions with line numbers
- Import statements
- Docstrings
- Complexity metrics
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code structure. Extracts function/class definitions, "
            "imports, docstrings, and basic complexity metrics from Python files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file to analyze.",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["overview", "functions", "classes", "imports", "metrics"],
                    "description": "Type of analysis to perform.",
                },
            },
            "required": ["path", "analysis_type"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for analysis."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_root(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    try:
        Path(path).resolve().relative_to(Path(_ALLOWED_ROOT).resolve())
        return True
    except ValueError:
        return False


def tool_function(
    path: str,
    analysis_type: str,
) -> str:
    """Analyze Python code structure.

    Args:
        path: Absolute path to Python file
        analysis_type: Type of analysis (overview, functions, classes, imports, metrics)

    Returns:
        Analysis results as formatted string
    """
    if not _is_within_root(path):
        return f"Error: Path '{path}' is outside allowed root."

    p = Path(path)
    if not p.is_absolute():
        return f"Error: {path} is not an absolute path."
    if not p.exists():
        return f"Error: {p} does not exist."
    if not p.is_file():
        return f"Error: {p} is not a file."
    if p.suffix != ".py":
        return f"Error: {p} is not a Python file."

    try:
        content = p.read_text()
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error: Syntax error in {p}: {e}"
    except Exception as e:
        return f"Error: Failed to parse {p}: {e}"

    if analysis_type == "overview":
        return _analyze_overview(tree, str(p))
    elif analysis_type == "functions":
        return _analyze_functions(tree, str(p))
    elif analysis_type == "classes":
        return _analyze_classes(tree, str(p))
    elif analysis_type == "imports":
        return _analyze_imports(tree, str(p))
    elif analysis_type == "metrics":
        return _analyze_metrics(tree, content, str(p))
    else:
        return f"Error: Unknown analysis_type '{analysis_type}'"


def _analyze_overview(tree: ast.AST, path: str) -> str:
    """Provide a high-level overview of the file."""
    imports = []
    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(_get_import_name(node))
        elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
            if not isinstance(getattr(node, "parent", None), ast.ClassDef):
                functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            if not isinstance(getattr(node, "parent", None), ast.ClassDef):
                functions.append(f"async {node.name}")
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)

    lines = [f"Overview of {path}:", ""]
    lines.append(f"Imports: {len(imports)}")
    lines.append(f"Top-level functions: {len(functions)}")
    lines.append(f"Classes: {len(classes)}")
    
    if imports:
        lines.extend(["", "Key imports:"])
        for imp in imports[:10]:
            lines.append(f"  - {imp}")
        if len(imports) > 10:
            lines.append(f"  ... and {len(imports) - 10} more")

    if functions:
        lines.extend(["", "Functions:"])
        for func in functions[:10]:
            lines.append(f"  - {func}")
        if len(functions) > 10:
            lines.append(f"  ... and {len(functions) - 10} more")

    if classes:
        lines.extend(["", "Classes:"])
        for cls in classes[:10]:
            lines.append(f"  - {cls}")
        if len(classes) > 10:
            lines.append(f"  ... and {len(classes) - 10} more")

    return "\n".join(lines)


def _analyze_functions(tree: ast.AST, path: str) -> str:
    """Extract detailed function information."""
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods (functions inside classes)
            if isinstance(getattr(node, "parent", None), ast.ClassDef):
                continue
            
            func_info = _get_function_info(node)
            functions.append(func_info)

    if not functions:
        return f"No top-level functions found in {path}"

    lines = [f"Functions in {path}:", ""]
    for func in functions:
        lines.append(f"Function: {func['name']} (line {func['line']})")
        lines.append(f"  Args: {', '.join(func['args']) if func['args'] else 'None'}")
        lines.append(f"  Returns: {func['returns'] or 'Not annotated'}")
        lines.append(f"  Docstring: {func['docstring'] or 'None'}")
        lines.append(f"  Complexity: {func['complexity']} branches")
        lines.append("")

    return "\n".join(lines)


def _analyze_classes(tree: ast.AST, path: str) -> str:
    """Extract detailed class information."""
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info = _get_class_info(node)
            classes.append(class_info)

    if not classes:
        return f"No classes found in {path}"

    lines = [f"Classes in {path}:", ""]
    for cls in classes:
        lines.append(f"Class: {cls['name']} (line {cls['line']})")
        if cls['bases']:
            lines.append(f"  Inherits: {', '.join(cls['bases'])}")
        lines.append(f"  Docstring: {cls['docstring'] or 'None'}")
        lines.append(f"  Methods: {len(cls['methods'])}")
        for method in cls['methods'][:5]:
            lines.append(f"    - {method['name']}({', '.join(method['args'])})")
        if len(cls['methods']) > 5:
            lines.append(f"    ... and {len(cls['methods']) - 5} more methods")
        lines.append("")

    return "\n".join(lines)


def _analyze_imports(tree: ast.AST, path: str) -> str:
    """Extract and categorize imports."""
    stdlib_imports = []
    third_party_imports = []
    local_imports = []

    stdlib_modules = {
        'os', 'sys', 'pathlib', 'typing', 'json', 're', 'subprocess', 'time',
        'datetime', 'collections', 'itertools', 'functools', 'math', 'random',
        'string', 'hashlib', 'logging', 'threading', 'multiprocessing', 'unittest',
        'abc', 'enum', 'dataclasses', 'contextlib', 'inspect', 'ast', 'types',
        'io', 'csv', 'xml', 'html', 'http', 'urllib', 'socket', 'email',
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split('.')[0]
                if name in stdlib_modules:
                    stdlib_imports.append(alias.name)
                else:
                    third_party_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            base = module.split('.')[0] if module else ""
            if base in stdlib_modules:
                stdlib_imports.append(f"from {module} import ...")
            elif node.level > 0:  # Relative import
                local_imports.append(f"from {module} import ...")
            else:
                third_party_imports.append(f"from {module} import ...")

    lines = [f"Imports in {path}:", ""]
    
    if stdlib_imports:
        lines.append("Standard library:")
        for imp in sorted(set(stdlib_imports)):
            lines.append(f"  - {imp}")
        lines.append("")
    
    if third_party_imports:
        lines.append("Third-party:")
        for imp in sorted(set(third_party_imports)):
            lines.append(f"  - {imp}")
        lines.append("")
    
    if local_imports:
        lines.append("Local/relative:")
        for imp in sorted(set(local_imports)):
            lines.append(f"  - {imp}")

    return "\n".join(lines)


def _analyze_metrics(tree: ast.AST, content: str, path: str) -> str:
    """Calculate basic code metrics."""
    lines = content.split('\n')
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    blank_lines = len([l for l in lines if not l.strip()])
    comment_lines = len([l for l in lines if l.strip().startswith('#')])

    # Count various constructs
    functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
    async_functions = len([n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)])
    classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
    
    # Calculate cyclomatic complexity approximation
    branches = len([n for n in ast.walk(tree) if isinstance(n, 
        (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With, ast.Assert,
         ast.comprehension, ast.BoolOp))])

    lines_out = [f"Metrics for {path}:", ""]
    lines_out.append(f"Total lines: {total_lines}")
    lines_out.append(f"Code lines: {code_lines}")
    lines_out.append(f"Blank lines: {blank_lines}")
    lines_out.append(f"Comment lines: {comment_lines}")
    lines_out.append(f"Functions: {functions}")
    lines_out.append(f"Async functions: {async_functions}")
    lines_out.append(f"Classes: {classes}")
    lines_out.append(f"Approximate complexity: {branches} branches")
    
    if total_lines > 0:
        lines_out.append(f"Code density: {code_lines/total_lines*100:.1f}%")

    return "\n".join(lines_out)


def _get_import_name(node: ast.Import | ast.ImportFrom) -> str:
    """Get a string representation of an import."""
    if isinstance(node, ast.Import):
        return ", ".join(alias.name for alias in node.names)
    else:
        module = node.module or ""
        names = ", ".join(alias.name for alias in node.names)
        return f"from {module} import {names}"


def _get_function_info(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Extract information about a function."""
    args = []
    for arg in node.args.args:
        args.append(arg.arg)
    for arg in node.args.kwonlyargs:
        args.append(arg.arg)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")

    returns = None
    if node.returns:
        returns = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

    docstring = ast.get_docstring(node)
    
    # Count branches for complexity
    branches = len([n for n in ast.walk(node) if isinstance(n, 
        (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With, ast.Assert,
         ast.comprehension, ast.BoolOp))])

    return {
        'name': node.name,
        'line': node.lineno,
        'args': args,
        'returns': returns,
        'docstring': docstring,
        'complexity': branches,
    }


def _get_class_info(node: ast.ClassDef) -> dict[str, Any]:
    """Extract information about a class."""
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))

    docstring = ast.get_docstring(node)

    methods = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_info = _get_function_info(item)
            methods.append(method_info)

    return {
        'name': node.name,
        'line': node.lineno,
        'bases': bases,
        'docstring': docstring,
        'methods': methods,
    }
