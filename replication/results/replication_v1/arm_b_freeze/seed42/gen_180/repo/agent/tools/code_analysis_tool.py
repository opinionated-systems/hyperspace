"""
Code analysis tool: analyze Python code for complexity, structure, and patterns.

Provides insights about code complexity, function/class counts, and potential issues.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "code_analysis",
            "description": "Analyze Python code for complexity metrics, structure, and patterns. Returns cyclomatic complexity, function/class counts, imports, and potential code smells.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to Python file or directory to analyze",
                    },
                    "include_metrics": {
                        "type": "boolean",
                        "description": "Include complexity metrics (default: true)",
                    },
                },
                "required": ["path"],
            },
        },
    }


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity for a function/method."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


def _analyze_file(file_path: str) -> dict[str, Any]:
    """Analyze a single Python file."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error in {file_path}: {e}"}
    except Exception as e:
        return {"error": f"Failed to parse {file_path}: {e}"}
    
    result = {
        "file": file_path,
        "lines": len(content.splitlines()),
        "characters": len(content),
        "functions": [],
        "classes": [],
        "imports": [],
        "complexity": 0,
    }
    
    total_complexity = 0
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "line": node.lineno,
                "args": len(node.args.args) + len(node.args.kwonlyargs),
                "complexity": _calculate_complexity(node),
            }
            result["functions"].append(func_info)
            total_complexity += func_info["complexity"]
        
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body 
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            result["classes"].append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "method_count": len(methods),
            })
        
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                module = node.module or ""
                names = [f"{module}.{alias.name}" if alias.name != "*" else f"{module}.*" 
                        for alias in node.names]
            result["imports"].extend(names)
    
    result["complexity"] = total_complexity
    result["function_count"] = len(result["functions"])
    result["class_count"] = len(result["classes"])
    result["import_count"] = len(result["imports"])
    
    # Detect potential issues
    issues = []
    
    # Check for long functions
    for func in result["functions"]:
        if func["complexity"] > 10:
            issues.append(f"Function '{func['name']}' has high complexity ({func['complexity']})")
        if func["args"] > 5:
            issues.append(f"Function '{func['name']}' has many arguments ({func['args']})")
    
    # Check for long classes
    for cls in result["classes"]:
        if cls["method_count"] > 20:
            issues.append(f"Class '{cls['name']}' has many methods ({cls['method_count']})")
    
    # Check for bare except clauses
    bare_except_pattern = r"except\s*:"
    if re.search(bare_except_pattern, content):
        issues.append("File contains bare 'except:' clauses")
    
    result["issues"] = issues
    result["issue_count"] = len(issues)
    
    return result


def tool_function(path: str, include_metrics: bool = True) -> str:
    """Analyze Python code and return formatted results."""
    target = Path(path)
    
    if not target.exists():
        return f"Error: Path not found: {path}"
    
    results = []
    
    if target.is_file():
        if not path.endswith(".py"):
            return f"Error: Not a Python file: {path}"
        results.append(_analyze_file(str(target)))
    else:
        # Analyze directory
        py_files = list(target.rglob("*.py"))
        if not py_files:
            return f"No Python files found in: {path}"
        
        for py_file in py_files:
            # Skip __pycache__ and hidden files
            if "__pycache__" in str(py_file) or py_file.name.startswith("."):
                continue
            file_result = _analyze_file(str(py_file))
            results.append(file_result)
    
    # Format output
    output_lines = []
    total_complexity = 0
    total_issues = 0
    
    for r in results:
        if "error" in r:
            output_lines.append(f"\n❌ {r['error']}")
            continue
        
        output_lines.append(f"\n📄 {r['file']}")
        output_lines.append(f"   Lines: {r['lines']}, Functions: {r['function_count']}, Classes: {r['class_count']}")
        
        if include_metrics:
            output_lines.append(f"   Cyclomatic Complexity: {r['complexity']}")
            total_complexity += r["complexity"]
        
        if r["functions"]:
            output_lines.append(f"   Functions:")
            for func in r["functions"]:
                complexity_info = f" (complexity: {func['complexity']})" if include_metrics else ""
                output_lines.append(f"     - {func['name']}(){complexity_info} at line {func['line']}")
        
        if r["classes"]:
            output_lines.append(f"   Classes:")
            for cls in r["classes"]:
                output_lines.append(f"     - {cls['name']} ({cls['method_count']} methods) at line {cls['line']}")
        
        if r["issues"]:
            output_lines.append(f"   ⚠️  Issues found:")
            for issue in r["issues"]:
                output_lines.append(f"      - {issue}")
            total_issues += r["issue_count"]
    
    # Summary
    output_lines.append(f"\n{'='*50}")
    output_lines.append(f"Summary: {len(results)} file(s) analyzed")
    if include_metrics:
        output_lines.append(f"Total cyclomatic complexity: {total_complexity}")
    output_lines.append(f"Total issues found: {total_issues}")
    
    return "\n".join(output_lines)
