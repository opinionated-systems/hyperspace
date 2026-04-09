"""
Code metrics tool for analyzing code quality and complexity.

Provides insights into code structure, complexity metrics, and potential issues.
Useful for understanding codebase health and identifying areas for improvement.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "analyze_code",
        "description": (
            "Analyze Python code for metrics like complexity, docstring coverage, "
            "and code structure. Returns detailed analysis of code quality indicators. "
            "Useful for understanding codebase health and identifying improvement areas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file or directory to analyze",
                },
                "include_metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to include: complexity, docstrings, imports, functions, classes",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to analyze in directory mode (default: 20)",
                },
            },
            "required": ["path"],
        },
    }


def _count_lines(content: str) -> dict[str, int]:
    """Count different types of lines in code."""
    lines = content.split("\n")
    total = len(lines)
    blank = sum(1 for line in lines if not line.strip())
    comment = sum(1 for line in lines if line.strip().startswith("#"))
    code = total - blank - comment
    return {"total": total, "code": code, "blank": blank, "comment": comment}


def _analyze_complexity(content: str) -> dict[str, Any]:
    """Analyze code complexity using simple heuristics."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    complexity = {
        "functions": 0,
        "classes": 0,
        "methods": 0,
        "branches": 0,
        "loops": 0,
        "max_function_lines": 0,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                complexity["functions"] += 1
            else:
                complexity["methods"] += 1
            # Estimate function length
            func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
            complexity["max_function_lines"] = max(complexity["max_function_lines"], func_lines)
        elif isinstance(node, ast.ClassDef):
            complexity["classes"] += 1
        elif isinstance(node, (ast.If, ast.IfExp)):
            complexity["branches"] += 1
        elif isinstance(node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp)):
            complexity["loops"] += 1

    return complexity


def _analyze_docstrings(content: str) -> dict[str, Any]:
    """Analyze docstring coverage."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"error": "Syntax error"}

    docstring_info = {
        "module_docstring": False,
        "functions_with_docstrings": 0,
        "functions_without_docstrings": 0,
        "classes_with_docstrings": 0,
        "classes_without_docstrings": 0,
    }

    # Check module docstring
    if tree.body and isinstance(tree.body[0], ast.Expr):
        if isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
            docstring_info["module_docstring"] = True

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            has_docstring = (
                ast.get_docstring(node) is not None
            )
            if has_docstring:
                docstring_info["functions_with_docstrings"] += 1
            else:
                docstring_info["functions_without_docstrings"] += 1
        elif isinstance(node, ast.ClassDef):
            has_docstring = ast.get_docstring(node) is not None
            if has_docstring:
                docstring_info["classes_with_docstrings"] += 1
            else:
                docstring_info["classes_without_docstrings"] += 1

    total_functions = docstring_info["functions_with_docstrings"] + docstring_info["functions_without_docstrings"]
    total_classes = docstring_info["classes_with_docstrings"] + docstring_info["classes_without_docstrings"]

    docstring_info["function_coverage"] = (
        f"{docstring_info['functions_with_docstrings']}/{total_functions}"
        if total_functions > 0 else "N/A"
    )
    docstring_info["class_coverage"] = (
        f"{docstring_info['classes_with_docstrings']}/{total_classes}"
        if total_classes > 0 else "N/A"
    )

    return docstring_info


def _analyze_imports(content: str) -> dict[str, Any]:
    """Analyze import statements."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"error": "Syntax error"}

    imports = {
        "standard_library": [],
        "third_party": [],
        "local": [],
    }

    stdlib_modules = {
        "os", "sys", "json", "re", "time", "datetime", "collections", "itertools",
        "functools", "pathlib", "typing", "inspect", "hashlib", "random", "math",
        "statistics", "string", "textwrap", "csv", "io", "pickle", "copy", "enum"
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                if name in stdlib_modules:
                    imports["standard_library"].append(name)
                elif name.startswith("."):
                    imports["local"].append(name)
                else:
                    imports["third_party"].append(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split(".")[0]
                if name in stdlib_modules:
                    imports["standard_library"].append(name)
                elif node.level > 0:  # Relative import
                    imports["local"].append(name)
                else:
                    imports["third_party"].append(name)

    # Remove duplicates while preserving order
    for key in imports:
        seen = set()
        unique = []
        for item in imports[key]:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        imports[key] = unique

    return imports


def _analyze_file(filepath: str, include_metrics: list[str] | None = None) -> dict[str, Any]:
    """Analyze a single Python file."""
    if include_metrics is None:
        include_metrics = ["complexity", "docstrings", "imports", "lines"]

    try:
        content = Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    result = {"file": filepath}

    if "lines" in include_metrics:
        result["lines"] = _count_lines(content)

    if "complexity" in include_metrics:
        result["complexity"] = _analyze_complexity(content)

    if "docstrings" in include_metrics:
        result["docstrings"] = _analyze_docstrings(content)

    if "imports" in include_metrics:
        result["imports"] = _analyze_imports(content)

    return result


def _format_analysis(analysis: dict[str, Any]) -> str:
    """Format analysis results as readable text."""
    lines = []
    lines.append(f"Analysis of: {analysis.get('file', 'unknown')}")
    lines.append("=" * 50)

    if "error" in analysis:
        lines.append(f"Error: {analysis['error']}")
        return "\n".join(lines)

    if "lines" in analysis:
        lines.append("\n📊 Line Counts:")
        for key, value in analysis["lines"].items():
            lines.append(f"  {key}: {value}")

    if "complexity" in analysis:
        lines.append("\n🔍 Complexity Metrics:")
        comp = analysis["complexity"]
        if "error" in comp:
            lines.append(f"  Error: {comp['error']}")
        else:
            for key, value in comp.items():
                lines.append(f"  {key}: {value}")

    if "docstrings" in analysis:
        lines.append("\n📝 Docstring Coverage:")
        docs = analysis["docstrings"]
        if "error" in docs:
            lines.append(f"  Error: {docs['error']}")
        else:
            lines.append(f"  Module docstring: {'✓' if docs['module_docstring'] else '✗'}")
            lines.append(f"  Function coverage: {docs['function_coverage']}")
            lines.append(f"  Class coverage: {docs['class_coverage']}")

    if "imports" in analysis:
        lines.append("\n📦 Imports:")
        imports = analysis["imports"]
        if "error" in imports:
            lines.append(f"  Error: {imports['error']}")
        else:
            for category, items in imports.items():
                if items:
                    lines.append(f"  {category}: {', '.join(items[:10])}")
                    if len(items) > 10:
                        lines.append(f"    ... and {len(items) - 10} more")

    return "\n".join(lines)


def tool_function(
    path: str,
    include_metrics: list[str] | None = None,
    max_files: int = 20,
) -> str:
    """Analyze Python code for metrics and quality indicators.

    Args:
        path: Path to Python file or directory to analyze
        include_metrics: List of metrics to include (complexity, docstrings, imports, lines)
        max_files: Maximum number of files to analyze in directory mode

    Returns:
        String with formatted analysis results
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    if include_metrics is None:
        include_metrics = ["complexity", "docstrings", "imports", "lines"]

    if os.path.isfile(path):
        if not path.endswith(".py"):
            return f"Error: File '{path}' is not a Python file"
        analysis = _analyze_file(path, include_metrics)
        return _format_analysis(analysis)

    # Directory mode
    results = []
    files_analyzed = 0

    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                analysis = _analyze_file(filepath, include_metrics)
                results.append(_format_analysis(analysis))
                files_analyzed += 1

                if files_analyzed >= max_files:
                    results.append(f"\n... (stopped after {max_files} files)")
                    break
        if files_analyzed >= max_files:
            break

    if not results:
        return f"No Python files found in '{path}'"

    return "\n\n" + "-" * 50 + "\n\n".join(results)
