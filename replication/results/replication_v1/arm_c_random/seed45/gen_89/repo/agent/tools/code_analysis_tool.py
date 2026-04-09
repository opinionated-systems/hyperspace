"""
Code analysis tool for identifying common Python code issues.

Provides static analysis capabilities to detect:
- Syntax errors
- Unused imports
- Undefined variables
- Basic code style issues
"""

from __future__ import annotations

import ast
import re
from typing import Any


def analyze_code(code: str, filename: str = "<unknown>") -> dict[str, Any]:
    """Analyze Python code for common issues.

    Args:
        code: Python source code to analyze
        filename: Name of the file (for error reporting)

    Returns:
        Dictionary with analysis results including:
        - syntax_valid: bool
        - syntax_errors: list of error messages
        - unused_imports: list of unused imported names
        - undefined_names: list of potentially undefined names
        - function_count: number of functions defined
        - class_count: number of classes defined
        - line_count: total lines of code
    """
    result = {
        "syntax_valid": True,
        "syntax_errors": [],
        "unused_imports": [],
        "undefined_names": [],
        "function_count": 0,
        "class_count": 0,
        "line_count": len(code.splitlines()),
    }

    # Check syntax
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as e:
        result["syntax_valid"] = False
        result["syntax_errors"].append(f"Line {e.lineno}: {e.msg}")
        return result
    except Exception as e:
        result["syntax_valid"] = False
        result["syntax_errors"].append(str(e))
        return result

    # Collect imports and their usage
    imports: dict[str, str] = {}  # name -> import node
    imported_names: set[str] = set()
    used_names: set[str] = set()
    defined_names: set[str] = set()

    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = alias.name
                imported_names.add(name)
                if alias.name != name:
                    imported_names.add(alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = f"{module}.{alias.name}"
                imported_names.add(name)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            defined_names.add(node.name)
            result["function_count"] += 1
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            defined_names.add(node.name)
            result["function_count"] += 1
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            defined_names.add(node.name)
            result["class_count"] += 1
            self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> None:
            defined_names.add(node.arg)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            # Add exception variable to defined names BEFORE visiting children
            if node.name:
                defined_names.add(node.name)
            # Visit type first (if any), then body
            if node.type:
                self.visit(node.type)
            for stmt in node.body:
                self.visit(stmt)

    visitor = ImportVisitor()
    visitor.visit(tree)

    # Find unused imports
    for name in imported_names:
        if name not in used_names and not name.startswith("_"):
            # Skip __future__ imports (always "used" for type annotations)
            if name in imports and imports[name] == "__future__":
                continue
            # Check if it's a module import that might be used for side effects
            if name in imports:
                result["unused_imports"].append(name)

    # Find potentially undefined names (heuristic)
    # Build comprehensive set of valid names
    builtin_names = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__.__class__))
    # Add common type names and builtin functions that might be used
    type_names = {"str", "int", "float", "bool", "list", "dict", "tuple", "set", "frozenset",
                  "type", "object", "Exception", "ValueError", "RuntimeError", "TypeError",
                  "KeyError", "IndexError", "AttributeError", "ImportError", "ModuleNotFoundError",
                  "IOError", "FileNotFoundError", "NotImplementedError", "StopIteration",
                  "GeneratorExit", "SystemExit", "KeyboardInterrupt", "ArithmeticError",
                  "LookupError", "AssertionError", "BufferError", "EOFError", "MemoryError",
                  "NameError", "OSError", "ReferenceError", "SyntaxError", "SystemError",
                  "RecursionError", "UnicodeError", "Warning", "UserWarning", "DeprecationWarning",
                  # Common builtin functions
                  "len", "any", "all", "sum", "min", "max", "abs", "round", "divmod",
                  "pow", "range", "enumerate", "zip", "map", "filter", "sorted",
                  "reversed", "iter", "next", "slice", "hasattr", "getattr", "setattr",
                  "delattr", "isinstance", "issubclass", "callable", "staticmethod",
                  "classmethod", "property", "super", "open", "input", "repr", "format",
                  "vars", "locals", "globals", "dir", "help", "id", "hash", "hex", "oct",
                  "bin", "chr", "ord", "ascii", "repr", "str", "bytes", "bytearray",
                  "memoryview", "compile", "eval", "exec", "breakpoint"}
    valid_names = defined_names | imported_names | builtin_names | type_names
    
    for name in used_names:
        if name not in valid_names:
            # Skip common patterns
            if not name.startswith("__") and not name.endswith("__"):
                result["undefined_names"].append(name)

    return result


def analyze_file(path: str) -> str:
    """Analyze a Python file and return a formatted report.

    Args:
        path: Path to the Python file

    Returns:
        Formatted analysis report as a string
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    result = analyze_code(code, filename=path)

    lines = [f"Analysis of {path}:", "=" * 50]

    if not result["syntax_valid"]:
        lines.append("\n❌ SYNTAX ERRORS:")
        for err in result["syntax_errors"]:
            lines.append(f"  - {err}")
        return "\n".join(lines)

    lines.append(f"\n✓ Syntax: Valid")
    lines.append(f"  Lines: {result['line_count']}")
    lines.append(f"  Functions: {result['function_count']}")
    lines.append(f"  Classes: {result['class_count']}")

    if result["unused_imports"]:
        lines.append(f"\n⚠ Unused imports ({len(result['unused_imports'])}):")
        for imp in result["unused_imports"]:
            lines.append(f"  - {imp}")
    else:
        lines.append("\n✓ No unused imports")

    if result["undefined_names"]:
        lines.append(f"\n⚠ Potentially undefined names ({len(result['undefined_names'])}):")
        for name in result["undefined_names"][:10]:  # Limit output
            lines.append(f"  - {name}")
        if len(result["undefined_names"]) > 10:
            lines.append(f"  ... and {len(result['undefined_names']) - 10} more")
    else:
        lines.append("\n✓ No undefined names detected")

    return "\n".join(lines)


def tool_info() -> dict:
    """Return tool registration info."""
    return {
        "name": "analyze_code",
        "description": "Analyze Python code for syntax errors, unused imports, undefined variables, and basic metrics. Can analyze code strings or files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to analyze (provide either code or path)",
                },
                "path": {
                    "type": "string",
                    "description": "Path to Python file to analyze (provide either code or path)",
                },
            },
        },
    }


def tool_function(code: str | None = None, path: str | None = None) -> str:
    """Tool entry point for code analysis.

    Args:
        code: Python source code string
        path: Path to Python file

    Returns:
        Analysis report as a string
    """
    if path:
        return analyze_file(path)
    elif code:
        result = analyze_code(code)

        lines = ["Code Analysis Results:", "=" * 40]

        if not result["syntax_valid"]:
            lines.append("\n❌ SYNTAX ERRORS:")
            for err in result["syntax_errors"]:
                lines.append(f"  - {err}")
            return "\n".join(lines)

        lines.append(f"\n✓ Syntax: Valid")
        lines.append(f"  Lines: {result['line_count']}")
        lines.append(f"  Functions: {result['function_count']}")
        lines.append(f"  Classes: {result['class_count']}")

        if result["unused_imports"]:
            lines.append(f"\n⚠ Unused imports ({len(result['unused_imports'])}):")
            for imp in result["unused_imports"]:
                lines.append(f"  - {imp}")

        if result["undefined_names"]:
            lines.append(f"\n⚠ Potentially undefined names ({len(result['undefined_names'])}):")
            for name in result["undefined_names"][:10]:
                lines.append(f"  - {name}")
            if len(result["undefined_names"]) > 10:
                lines.append(f"  ... and {len(result['undefined_names']) - 10} more")

        return "\n".join(lines)
    else:
        return "Error: Provide either 'code' or 'path' parameter"
