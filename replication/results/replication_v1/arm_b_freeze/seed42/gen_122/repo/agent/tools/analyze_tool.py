"""
Code analysis tool for understanding code structure and complexity.

Provides insights into Python files including:
- Function/class counts
- Complexity metrics
- Dependency analysis
- Code quality indicators
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for the analyze tool."""
    return {
        "name": "analyze_code",
        "description": "Analyze Python code structure, complexity, and metrics. Provides insights into functions, classes, imports, and code quality indicators.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the Python file to analyze",
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Whether to include complexity metrics (default: true)",
                },
            },
            "required": ["file_path"],
        },
    }


def _count_function_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity for a function."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                              ast.With, ast.Assert, ast.comprehension)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


def _analyze_file(file_path: str, include_metrics: bool = True) -> dict[str, Any]:
    """Analyze a Python file and return structured information."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    if not file_path.endswith('.py'):
        return {"error": f"Not a Python file: {file_path}"}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"error": f"Syntax error in file: {e}"}
    
    # Basic metrics
    lines = source.split('\n')
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    
    result: dict[str, Any] = {
        "file_path": file_path,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "blank_lines": total_lines - code_lines - len([l for l in lines if l.strip().startswith('#')]),
    }
    
    # AST analysis
    functions = []
    classes = []
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "line": node.lineno,
                "args": len(node.args.args),
                "docstring": ast.get_docstring(node) is not None,
            }
            if include_metrics:
                func_info["complexity"] = _count_function_complexity(node)
            functions.append(func_info)
        
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "line": node.lineno,
                "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                "docstring": ast.get_docstring(node) is not None,
            }
            classes.append(class_info)
        
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"type": "import", "name": alias.name})
            else:
                module = node.module or ""
                for alias in node.names:
                    imports.append({"type": "from", "module": module, "name": alias.name})
    
    result["functions"] = functions
    result["classes"] = classes
    result["imports"] = imports[:20]  # Limit imports to avoid overwhelming output
    result["function_count"] = len(functions)
    result["class_count"] = len(classes)
    result["import_count"] = len(imports)
    
    # Quality indicators
    if include_metrics:
        avg_complexity = sum(f.get("complexity", 1) for f in functions) / max(len(functions), 1)
        result["metrics"] = {
            "average_complexity": round(avg_complexity, 2),
            "high_complexity_functions": [f["name"] for f in functions if f.get("complexity", 1) > 5],
            "undocumented_functions": [f["name"] for f in functions if not f["docstring"]],
            "undocumented_classes": [c["name"] for c in classes if not c["docstring"]],
        }
    
    return result


def tool_function(file_path: str, include_metrics: bool = True) -> str:
    """Analyze a Python file and return structured information.
    
    Args:
        file_path: Absolute path to the Python file to analyze
        include_metrics: Whether to include complexity metrics (default: true)
    
    Returns:
        JSON string with analysis results
    """
    result = _analyze_file(file_path, include_metrics)
    return json.dumps(result, indent=2)
