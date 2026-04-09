"""
Code analysis tool: analyze Python code structure and extract useful information.

Provides functionality to:
- Extract function/class definitions
- Count lines of code
- Identify imports
- Analyze code complexity
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code structure. "
            "Extracts functions, classes, imports, and metrics. "
            "Useful for understanding code before modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file.",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["summary", "functions", "classes", "imports", "metrics"],
                    "description": "Type of analysis to perform.",
                },
            },
            "required": ["path", "analysis_type"],
        },
    }


def tool_function(path: str, analysis_type: str) -> str:
    """Execute code analysis on a Python file."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {p} does not exist."

        if not p.is_file():
            return f"Error: {p} is not a file."

        content = p.read_text()

        if analysis_type == "summary":
            return _analyze_summary(content, str(p))
        elif analysis_type == "functions":
            return _analyze_functions(content)
        elif analysis_type == "classes":
            return _analyze_classes(content)
        elif analysis_type == "imports":
            return _analyze_imports(content)
        elif analysis_type == "metrics":
            return _analyze_metrics(content)
        else:
            return f"Error: unknown analysis_type {analysis_type}"
    except Exception as e:
        return f"Error: {e}"


def _analyze_summary(content: str, path: str) -> str:
    """Provide a summary of the Python file."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Syntax error in file: {e}"

    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

    lines = content.split("\n")
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    comment_lines = len([l for l in lines if l.strip().startswith("#")])
    blank_lines = len([l for l in lines if not l.strip()])

    summary = [
        f"File: {path}",
        f"Total lines: {len(lines)}",
        f"Code lines: {code_lines}",
        f"Comment lines: {comment_lines}",
        f"Blank lines: {blank_lines}",
        f"Functions: {len(functions)}",
        f"Classes: {len(classes)}",
        f"Imports: {len(imports)}",
    ]

    return "\n".join(summary)


def _analyze_functions(content: str) -> str:
    """Extract function definitions from the code."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Syntax error in file: {e}"

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            functions.append(f"- {node.name}({', '.join(args)}) at line {node.lineno}")

    if not functions:
        return "No functions found."

    return "Functions:\n" + "\n".join(functions)


def _analyze_classes(content: str) -> str:
    """Extract class definitions from the code."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Syntax error in file: {e}"

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes.append(f"- {node.name}{base_str} at line {node.lineno}")
            if methods:
                classes.append(f"  Methods: {', '.join(methods)}")

    if not classes:
        return "No classes found."

    return "Classes:\n" + "\n".join(classes)


def _analyze_imports(content: str) -> str:
    """Extract imports from the code."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Syntax error in file: {e}"

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"- import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(f"- from {module} import {', '.join(names)}")

    if not imports:
        return "No imports found."

    return "Imports:\n" + "\n".join(imports)


def _analyze_metrics(content: str) -> str:
    """Calculate code metrics."""
    lines = content.split("\n")

    # Basic metrics
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    comment_lines = len([l for l in lines if l.strip().startswith("#")])
    blank_lines = len([l for l in lines if not l.strip()])

    # Try to calculate cyclomatic complexity approximation
    complexity_indicators = [
        r"\bif\b", r"\belif\b", r"\belse\b",
        r"\bfor\b", r"\bwhile\b",
        r"\bexcept\b", r"\bfinally\b",
        r"\band\b", r"\bor\b",
        r"\bassert\b",
    ]

    complexity_score = 0
    for line in lines:
        for pattern in complexity_indicators:
            complexity_score += len(re.findall(pattern, line))

    metrics = [
        "Code Metrics:",
        f"- Total lines: {total_lines}",
        f"- Code lines: {code_lines}",
        f"- Comment lines: {comment_lines}",
        f"- Blank lines: {blank_lines}",
        f"- Comment ratio: {comment_lines / total_lines * 100:.1f}%" if total_lines > 0 else "- Comment ratio: 0%",
        f"- Approximate complexity score: {complexity_score}",
    ]

    return "\n".join(metrics)
