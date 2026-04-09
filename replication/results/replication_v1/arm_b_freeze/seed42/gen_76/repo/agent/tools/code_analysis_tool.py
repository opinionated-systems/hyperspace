"""
Code analysis tool for analyzing Python code structure and complexity.

Provides insights into code organization, imports, functions, and complexity metrics.
"""

from __future__ import annotations

import ast
import os
from typing import Any


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "analyze_code",
        "description": (
            "Analyze Python code structure and complexity. "
            "Returns information about imports, functions, classes, "
            "and complexity metrics. Useful for understanding code organization."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to Python file or directory to analyze",
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Whether to include complexity metrics (default: true)",
                },
            },
            "required": ["path"],
        },
    }


def _analyze_file(filepath: str, include_metrics: bool = True) -> dict[str, Any]:
    """Analyze a single Python file."""
    result = {
        "file": filepath,
        "imports": [],
        "functions": [],
        "classes": [],
        "lines": 0,
    }
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            result["lines"] = len(content.splitlines())
    except Exception as e:
        return {"file": filepath, "error": str(e)}
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"file": filepath, "error": f"Syntax error: {e}"}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            result["imports"].append(f"{module}: {', '.join(names)}")
        elif isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "line": node.lineno,
                "args": len(node.args.args),
            }
            if include_metrics:
                # Simple complexity: count of statements in function body
                stmt_count = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                func_info["statements"] = stmt_count
            result["functions"].append(func_info)
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "line": node.lineno,
                "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
            }
            result["classes"].append(class_info)
    
    return result


def tool_function(path: str, include_metrics: bool = True) -> str:
    """Analyze Python code structure and complexity.

    Args:
        path: Path to Python file or directory to analyze
        include_metrics: Whether to include complexity metrics

    Returns:
        String with analysis results
    """
    if not os.path.exists(path):
        return f"Error: '{path}' does not exist"
    
    results = []
    
    if os.path.isfile(path):
        if not path.endswith(".py"):
            return f"Error: '{path}' is not a Python file"
        results.append(_analyze_file(path, include_metrics))
    else:
        # Analyze directory
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = os.path.join(root, filename)
                    results.append(_analyze_file(filepath, include_metrics))
    
    # Format output
    lines = [f"Code Analysis for: {path}", "=" * 50]
    
    total_lines = 0
    total_functions = 0
    total_classes = 0
    
    for r in results:
        if "error" in r:
            lines.append(f"\nFile: {r['file']}")
            lines.append(f"  Error: {r['error']}")
            continue
        
        lines.append(f"\nFile: {r['file']} ({r['lines']} lines)")
        total_lines += r["lines"]
        
        if r["imports"]:
            lines.append(f"  Imports: {len(r['imports'])}")
            for imp in r["imports"][:5]:  # Show first 5
                lines.append(f"    - {imp}")
            if len(r["imports"]) > 5:
                lines.append(f"    ... and {len(r['imports']) - 5} more")
        
        if r["functions"]:
            lines.append(f"  Functions: {len(r['functions'])}")
            total_functions += len(r["functions"])
            for func in r["functions"][:3]:  # Show first 3
                metrics = f", {func.get('statements', 0)} stmts" if include_metrics else ""
                lines.append(f"    - {func['name']}() (line {func['line']}{metrics})")
            if len(r["functions"]) > 3:
                lines.append(f"    ... and {len(r['functions']) - 3} more")
        
        if r["classes"]:
            lines.append(f"  Classes: {len(r['classes'])}")
            total_classes += len(r["classes"])
            for cls in r["classes"]:
                lines.append(f"    - {cls['name']} (line {cls['line']}, {cls['methods']} methods)")
    
    # Summary
    lines.append("\n" + "=" * 50)
    lines.append(f"Summary: {len(results)} files, {total_lines} lines, {total_functions} functions, {total_classes} classes")
    
    return "\n".join(lines)
