"""
Code analysis tool: analyze Python code structure and metrics.

Provides capabilities to understand code structure, find definitions,
analyze complexity, and extract useful information from Python files.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code structure and metrics. "
            "Provides insights into functions, classes, imports, complexity, and code patterns. "
            "Use this to understand code before making modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["analyze_file", "find_definitions", "get_metrics", "extract_imports", "detect_patterns"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Python file to analyze.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional query for find_definitions (e.g., function or class name).",
                },
            },
            "required": ["command", "path"],
        },
    }


def _count_lines(content: str) -> dict[str, int]:
    """Count total, code, blank, and comment lines."""
    lines = content.split("\n")
    total = len(lines)
    blank = sum(1 for line in lines if line.strip() == "")
    comment = sum(1 for line in lines if line.strip().startswith("#"))
    code = total - blank - comment
    return {"total": total, "code": code, "blank": blank, "comment": comment}


def _extract_imports(content: str) -> list[dict[str, Any]]:
    """Extract import statements from Python code."""
    imports = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    })
    except SyntaxError:
        pass
    return imports


def _find_definitions(content: str, query: str | None = None) -> list[dict[str, Any]]:
    """Find function and class definitions in Python code."""
    definitions = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if query is None or query in node.name:
                    # Get function signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)
                    
                    # Handle *args and **kwargs
                    if node.args.vararg:
                        args.append(f"*{node.args.vararg.arg}")
                    if node.args.kwarg:
                        args.append(f"**{node.args.kwarg.arg}")
                    
                    signature = f"{node.name}({', '.join(args)})"
                    definitions.append({
                        "type": "function",
                        "name": node.name,
                        "signature": signature,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node),
                    })
            elif isinstance(node, ast.ClassDef):
                if query is None or query in node.name:
                    # Get base classes
                    bases = [ast.unparse(base) for base in node.bases]
                    definitions.append({
                        "type": "class",
                        "name": node.name,
                        "bases": bases,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node),
                    })
    except SyntaxError:
        pass
    return definitions


def _calculate_complexity(content: str) -> dict[str, Any]:
    """Calculate basic complexity metrics for Python code."""
    try:
        tree = ast.parse(content)
        
        # Count different node types
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        
        # Calculate cyclomatic complexity (simplified)
        branches = len([n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.While, ast.For, ast.ExceptHandler))])
        branches += len([n for n in ast.walk(tree) if isinstance(n, ast.BoolOp)])
        
        # Count returns
        returns = len([n for n in ast.walk(tree) if isinstance(n, ast.Return)])
        
        return {
            "functions": function_count,
            "classes": class_count,
            "branches": branches,
            "returns": returns,
            "cyclomatic_complexity": branches + 1,
        }
    except SyntaxError:
        return {"error": "Could not parse file"}


def _detect_patterns(content: str) -> dict[str, Any]:
    """Detect common code patterns and potential issues."""
    patterns = {
        "error_handling": {
            "try_except_blocks": len(re.findall(r'\btry\b:', content)),
            "bare_excepts": len(re.findall(r'except\s*:', content)),
            "exception_types": len(re.findall(r'except\s+\w+', content)),
        },
        "logging": {
            "has_logging_import": bool(re.search(r'import\s+logging', content)),
            "logger_calls": len(re.findall(r'logger\.(debug|info|warning|error|critical)', content)),
            "print_statements": len(re.findall(r'\bprint\s*\(', content)),
        },
        "type_hints": {
            "function_annotations": len(re.findall(r'def\s+\w+\s*\([^)]*\)\s*->\s*\w+', content)),
            "arg_annotations": len(re.findall(r'\w+\s*:\s*\w+', content)),
            "total_functions": len(re.findall(r'\bdef\s+\w+', content)),
        },
        "documentation": {
            "docstrings": len(re.findall(r'"""[\s\S]*?"""', content)),
            "comments": len(re.findall(r'#.*', content)),
        },
        "code_style": {
            "classes": len(re.findall(r'\bclass\s+\w+', content)),
            "functions": len(re.findall(r'\bdef\s+\w+', content)),
            "list_comprehensions": len(re.findall(r'\[.*for.*in.*\]', content)),
            "generator_expressions": len(re.findall(r'\(.*for.*in.*\)', content)),
        },
    }
    
    # Calculate coverage metrics
    total_funcs = patterns["type_hints"]["total_functions"]
    annotated_funcs = patterns["type_hints"]["function_annotations"]
    patterns["type_hints"]["annotation_coverage"] = (
        f"{annotated_funcs}/{total_funcs} ({annotated_funcs/max(total_funcs, 1)*100:.1f}%)"
    )
    
    # Identify potential improvements
    suggestions = []
    if patterns["error_handling"]["bare_excepts"] > 0:
        suggestions.append(f"Replace {patterns['error_handling']['bare_excepts']} bare except clauses with specific exception types")
    if patterns["logging"]["print_statements"] > 0 and not patterns["logging"]["has_logging_import"]:
        suggestions.append(f"Consider replacing {patterns['logging']['print_statements']} print statements with logging")
    if patterns["type_hints"]["annotation_coverage"].startswith("0/"):
        suggestions.append("Consider adding type hints to functions")
    if patterns["documentation"]["docstrings"] < patterns["code_style"]["functions"]:
        suggestions.append("Add docstrings to undocumented functions")
    
    patterns["suggestions"] = suggestions
    return patterns


def _analyze_file(path: Path) -> dict[str, Any]:
    """Perform comprehensive file analysis."""
    if not path.exists():
        return {"error": f"File {path} does not exist"}
    
    if not path.suffix == ".py":
        return {"error": f"File {path} is not a Python file"}
    
    content = path.read_text()
    
    return {
        "path": str(path),
        "lines": _count_lines(content),
        "imports": _extract_imports(content),
        "definitions": _find_definitions(content),
        "complexity": _calculate_complexity(content),
        "patterns": _detect_patterns(content),
    }


def tool_function(
    command: str,
    path: str,
    query: str | None = None,
) -> str:
    """Execute a code analysis command.
    
    Args:
        command: The analysis command (analyze_file, find_definitions, get_metrics, extract_imports, detect_patterns)
        path: Absolute path to the Python file
        query: Optional query string for find_definitions
    
    Returns:
        JSON string with analysis results
    """
    import json
    
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return json.dumps({"error": f"Path {path} is not absolute"})
        
        if not p.exists():
            return json.dumps({"error": f"File {path} does not exist"})
        
        if p.suffix != ".py":
            return json.dumps({"error": f"File {path} is not a Python file"})
        
        content = p.read_text()
        
        if command == "analyze_file":
            result = _analyze_file(p)
        elif command == "find_definitions":
            result = {
                "path": str(p),
                "definitions": _find_definitions(content, query),
                "query": query,
            }
        elif command == "get_metrics":
            result = {
                "path": str(p),
                "lines": _count_lines(content),
                "complexity": _calculate_complexity(content),
            }
        elif command == "extract_imports":
            result = {
                "path": str(p),
                "imports": _extract_imports(content),
            }
        elif command == "detect_patterns":
            result = {
                "path": str(p),
                "patterns": _detect_patterns(content),
            }
        else:
            return json.dumps({"error": f"Unknown command: {command}"})
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": str(e)})
