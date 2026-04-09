"""
Code analysis tool: analyze Python code for complexity, style issues, and structure.

Provides insights into code quality metrics to guide improvements.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


def _count_lines(content: str) -> dict[str, int]:
    """Count different types of lines in code."""
    lines = content.splitlines()
    total = len(lines)
    code_lines = 0
    comment_lines = 0
    blank_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith('#'):
            comment_lines += 1
        else:
            code_lines += 1

    return {
        "total": total,
        "code": code_lines,
        "comments": comment_lines,
        "blank": blank_lines,
    }


def _analyze_complexity(content: str) -> dict[str, Any]:
    """Analyze code complexity using AST."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    functions = []
    classes = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Calculate cyclomatic complexity (simplified)
            complexity = 1
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                    ast.With, ast.Assert, ast.comprehension)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1

            functions.append({
                "name": node.name,
                "line": node.lineno,
                "complexity": complexity,
                "args": len(node.args.args),
            })
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "method_count": len(methods),
            })
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [f"{node.module}.{alias.name}" if node.module else alias.name
                        for alias in node.names]
            imports.extend(names)

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "function_count": len(functions),
        "class_count": len(classes),
    }


def _check_style_issues(content: str) -> list[dict]:
    """Check for common style issues."""
    issues = []
    lines = content.splitlines()

    for i, line in enumerate(lines, 1):
        # Check line length
        if len(line) > 100:
            issues.append({
                "line": i,
                "type": "long_line",
                "message": f"Line exceeds 100 characters ({len(line)} chars)",
            })

        # Check trailing whitespace
        if line.rstrip() != line:
            issues.append({
                "line": i,
                "type": "trailing_whitespace",
                "message": "Line has trailing whitespace",
            })

        # Check for TODO/FIXME comments
        if re.search(r'#.*\b(TODO|FIXME|XXX|HACK)\b', line, re.IGNORECASE):
            match = re.search(r'#.*\b(TODO|FIXME|XXX|HACK)\b', line, re.IGNORECASE)
            issues.append({
                "line": i,
                "type": "marker",
                "message": f"Found {match.group(1).upper()} marker",
            })

    return issues


def analyze_code(file_path: str) -> dict[str, Any]:
    """Analyze a Python file and return metrics.

    Args:
        file_path: Path to the Python file to analyze.

    Returns:
        Dictionary with analysis results including:
        - lines: Line counts (total, code, comments, blank)
        - complexity: Function/class analysis with complexity metrics
        - style: Style issues found
        - summary: Brief summary of the file
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    if not file_path.endswith('.py'):
        return {"error": "Only Python files (.py) are supported"}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    lines = _count_lines(content)
    complexity = _analyze_complexity(content)
    style = _check_style_issues(content)

    # Build summary
    summary_parts = []
    summary_parts.append(
        f"{lines['total']} lines ({lines['code']} code, {lines['comments']} comments)"
    )

    if complexity.get("function_count"):
        summary_parts.append(f"{complexity['function_count']} functions")
    if complexity.get("class_count"):
        summary_parts.append(f"{complexity['class_count']} classes")
    if style:
        summary_parts.append(f"{len(style)} style issues")

    return {
        "file": file_path,
        "lines": lines,
        "complexity": complexity,
        "style_issues": style,
        "summary": "; ".join(summary_parts),
    }


def tool_info() -> dict:
    """Return tool information for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "analyze_code",
            "description": (
                "Analyze Python code files for complexity, style issues, and structure. "
                "Returns metrics like line counts, function complexity, class structure, "
                "and style warnings. Useful for understanding code quality before modifications."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the Python file to analyze",
                    },
                },
                "required": ["file_path"],
            },
        },
    }


tool_function = analyze_code
