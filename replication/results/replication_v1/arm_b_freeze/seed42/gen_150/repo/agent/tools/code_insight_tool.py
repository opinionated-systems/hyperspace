"""
Code insight tool for analyzing Python code structure.

Provides code complexity metrics, function/class extraction, and dependency analysis.
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
            "Analyze Python code structure to extract functions, classes, imports, and complexity metrics. "
            "Useful for understanding code organization and identifying areas for improvement."
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
                    "description": "Include complexity metrics (lines, function count, etc.)",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to analyze in a directory (default: 20)",
                },
            },
            "required": ["path"],
        },
    }


def _analyze_file(filepath: str) -> dict[str, Any]:
    """Analyze a single Python file."""
    result = {
        "file": filepath,
        "functions": [],
        "classes": [],
        "imports": [],
        "metrics": {},
        "error": None,
    }
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        
        lines = source.splitlines()
        result["metrics"]["total_lines"] = len(lines)
        result["metrics"]["code_lines"] = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            result["error"] = f"Syntax error: {e}"
            return result
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": len(node.args.args) + len(node.args.kwonlyargs),
                    "docstring": ast.get_docstring(node),
                }
                result["functions"].append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    "docstring": ast.get_docstring(node),
                }
                result["classes"].append(class_info)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result["imports"].append(alias.name)
                else:
                    module = node.module or ""
                    for alias in node.names:
                        result["imports"].append(f"{module}.{alias.name}" if module else alias.name)
        
        result["metrics"]["function_count"] = len(result["functions"])
        result["metrics"]["class_count"] = len(result["classes"])
        result["metrics"]["import_count"] = len(result["imports"])
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def tool_function(
    path: str,
    include_metrics: bool = True,
    max_files: int = 20,
) -> str:
    """Analyze Python code structure.

    Args:
        path: Path to Python file or directory to analyze
        include_metrics: Include complexity metrics
        max_files: Maximum number of files to analyze in a directory

    Returns:
        String with analysis results
    """
    if not os.path.exists(path):
        return f"Error: '{path}' does not exist"
    
    files_to_analyze = []
    
    if os.path.isfile(path):
        if path.endswith(".py"):
            files_to_analyze.append(path)
        else:
            return f"Error: '{path}' is not a Python file"
    else:
        # Find Python files in directory
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".py"):
                    files_to_analyze.append(os.path.join(root, f))
                    if len(files_to_analyze) >= max_files:
                        break
            if len(files_to_analyze) >= max_files:
                break
    
    if not files_to_analyze:
        return "No Python files found to analyze."
    
    results = []
    for filepath in files_to_analyze:
        analysis = _analyze_file(filepath)
        
        lines = [f"📄 {analysis['file']}"]
        
        if analysis["error"]:
            lines.append(f"  ⚠️  {analysis['error']}")
            results.append("\n".join(lines))
            continue
        
        if include_metrics and analysis["metrics"]:
            m = analysis["metrics"]
            metrics_line = f"  📊 Lines: {m.get('code_lines', 0)}/{m.get('total_lines', 0)} code/total"
            if m.get("function_count"):
                metrics_line += f" | Functions: {m['function_count']}"
            if m.get("class_count"):
                metrics_line += f" | Classes: {m['class_count']}"
            lines.append(metrics_line)
        
        if analysis["classes"]:
            lines.append("  🏛️  Classes:")
            for cls in analysis["classes"]:
                doc = " (has docstring)" if cls["docstring"] else ""
                lines.append(f"    - {cls['name']} (line {cls['line']}, {cls['methods']} methods){doc}")
        
        if analysis["functions"]:
            lines.append("  🔧 Functions:")
            for func in analysis["functions"]:
                doc = " ✓" if func["docstring"] else ""
                lines.append(f"    - {func['name']}({func['args']} args) line {func['line']}{doc}")
        
        if analysis["imports"]:
            unique_imports = list(set(analysis["imports"]))[:10]  # Limit imports shown
            lines.append(f"  📦 Imports: {', '.join(unique_imports)}")
            if len(analysis["imports"]) > 10:
                lines.append(f"    ... and {len(analysis['imports']) - 10} more")
        
        results.append("\n".join(lines))
    
    header = f"Analyzed {len(files_to_analyze)} file(s):\n"
    if len(files_to_analyze) == max_files:
        header = f"Analyzed {len(files_to_analyze)} file(s) (max limit reached):\n"
    
    return header + "\n\n".join(results)
