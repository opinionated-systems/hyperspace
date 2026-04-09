"""
Code analysis tool: analyze Python code structure.

Provides capabilities to extract information about classes, functions,
imports, and other structural elements from Python source files.
"""

from __future__ import annotations

import ast
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "code_analysis",
        "description": "Analyze Python code structure to extract classes, functions, imports, and other structural elements. Useful for understanding code organization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The Python file path to analyze",
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform: 'all', 'functions', 'classes', 'imports', 'summary'",
                    "enum": ["all", "functions", "classes", "imports", "summary"],
                },
            },
            "required": ["path"],
        },
    }


def tool_function(
    path: str,
    analysis_type: str = "all",
) -> str:
    """Analyze Python code structure.

    Args:
        path: The Python file path to analyze
        analysis_type: Type of analysis to perform

    Returns:
        Analysis results as formatted text
    """
    try:
        with open(path, "r") as f:
            source = f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Error: Syntax error in file: {e}"

    results = []

    if analysis_type in ("all", "summary"):
        summary = _get_summary(tree)
        results.append("=== SUMMARY ===")
        results.append(summary)
        results.append("")

    if analysis_type in ("all", "imports"):
        imports = _get_imports(tree)
        if imports:
            results.append("=== IMPORTS ===")
            for imp in imports:
                results.append(f"  {imp}")
            results.append("")

    if analysis_type in ("all", "classes"):
        classes = _get_classes(tree)
        if classes:
            results.append("=== CLASSES ===")
            for cls in classes:
                results.append(f"  {cls}")
            results.append("")

    if analysis_type in ("all", "functions"):
        functions = _get_functions(tree)
        if functions:
            results.append("=== FUNCTIONS ===")
            for func in functions:
                results.append(f"  {func}")
            results.append("")

    return "\n".join(results) if results else "No results found."


def _get_summary(tree: ast.AST) -> str:
    """Get a summary of the Python file."""
    num_imports = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
    num_classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
    num_functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
    num_async = len([n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)])

    lines = [
        f"Total imports: {num_imports}",
        f"Total classes: {num_classes}",
        f"Total functions: {num_functions}",
        f"Total async functions: {num_async}",
    ]
    return "\n".join(lines)


def _get_imports(tree: ast.AST) -> list[str]:
    """Extract all imports from the AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(f"from {module} import {', '.join(names)}")
    return imports


def _get_classes(tree: ast.AST) -> list[str]:
    """Extract all class definitions from the AST."""
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else "...")

            # Get methods
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)

            base_str = f"({', '.join(bases)})" if bases else ""
            method_str = f" - methods: {', '.join(methods)}" if methods else ""
            classes.append(f"class {node.name}{base_str}{method_str}")
    return classes


def _get_functions(tree: ast.AST) -> list[str]:
    """Extract all function definitions from the AST."""
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods (functions inside classes)
            if isinstance(getattr(node, "parent", None), ast.ClassDef):
                continue

            # Get arguments
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            for arg in node.args.kwonlyargs:
                args.append(arg.arg)
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")

            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            functions.append(f"{prefix}def {node.name}({', '.join(args)})")
    return functions
