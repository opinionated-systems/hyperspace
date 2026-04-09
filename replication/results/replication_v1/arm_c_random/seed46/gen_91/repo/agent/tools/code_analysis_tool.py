"""
Code analysis tool: analyze Python code for common issues.

Provides static analysis capabilities to help identify potential
bugs, style issues, and improvement opportunities in Python code.
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
            "Analyze Python code for common issues, style problems, and potential bugs. "
            "Provides suggestions for improvements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Python file to analyze.",
                },
                "code": {
                    "type": "string",
                    "description": "Python code string to analyze (alternative to path).",
                },
            },
        },
    }


def _analyze_syntax(code: str) -> list[dict]:
    """Check for syntax errors."""
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append({
            "type": "syntax_error",
            "line": e.lineno,
            "message": str(e),
            "severity": "error"
        })
    return issues


def _analyze_imports(tree: ast.AST) -> list[dict]:
    """Check for import-related issues."""
    issues = []
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    # Check for unused imports would require more complex analysis
    # For now, just report the imports found
    return issues


def _analyze_functions(tree: ast.AST) -> list[dict]:
    """Check for function-related issues."""
    issues = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check for empty functions
            if not node.body or (len(node.body) == 1 and 
                                isinstance(node.body[0], ast.Pass)):
                issues.append({
                    "type": "empty_function",
                    "line": node.lineno,
                    "message": f"Function '{node.name}' is empty",
                    "severity": "warning"
                })
            
            # Check for very long functions
            func_end = node.end_lineno or node.lineno
            func_length = func_end - node.lineno
            if func_length > 100:
                issues.append({
                    "type": "long_function",
                    "line": node.lineno,
                    "message": f"Function '{node.name}' is {func_length} lines (consider refactoring)",
                    "severity": "info"
                })
            
            # Check for functions without docstrings
            has_docstring = False
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
                has_docstring = True
            
            if not has_docstring and not node.name.startswith("_"):
                issues.append({
                    "type": "missing_docstring",
                    "line": node.lineno,
                    "message": f"Function '{node.name}' lacks a docstring",
                    "severity": "info"
                })
    
    return issues


def _analyze_complexity(tree: ast.AST) -> list[dict]:
    """Check code complexity issues."""
    issues = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            # Check for bare except clauses
            for handler in node.handlers:
                if handler.type is None:
                    issues.append({
                        "type": "bare_except",
                        "line": handler.lineno,
                        "message": "Bare 'except:' clause - should catch specific exceptions",
                        "severity": "warning"
                    })
        
        if isinstance(node, ast.ExceptHandler):
            # Check for catching Exception or BaseException without good reason
            if isinstance(node.type, ast.Name):
                if node.type.id in ("Exception", "BaseException"):
                    # This is a warning but not always wrong
                    pass
    
    return issues


def _analyze_patterns(code: str) -> list[dict]:
    """Check for common problematic patterns."""
    issues = []
    lines = code.split("\n")
    
    for i, line in enumerate(lines, 1):
        # Check for mutable default arguments
        if re.search(r"def\s+\w+\s*\([^)]*=\s*(\[|\{)", line):
            issues.append({
                "type": "mutable_default",
                "line": i,
                "message": "Possible mutable default argument (list or dict)",
                "severity": "warning"
            })
        
        # Check for == None or != None (should be 'is None' or 'is not None')
        if re.search(r"==\s*None|!=\s*None", line):
            issues.append({
                "type": "none_comparison",
                "line": i,
                "message": "Use 'is None' or 'is not None' instead of == or !=",
                "severity": "warning"
            })
        
        # Check for very long lines
        if len(line) > 120:
            issues.append({
                "type": "long_line",
                "line": i,
                "message": f"Line is {len(line)} characters (consider breaking)",
                "severity": "info"
            })
    
    return issues


def tool_function(path: str | None = None, code: str | None = None) -> str:
    """Analyze Python code for common issues and improvements."""
    # Validate inputs
    if path and code:
        return "Error: Provide either 'path' or 'code', not both."
    
    if not path and not code:
        return "Error: Provide either 'path' or 'code'."
    
    # Get the code to analyze
    if path:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: File {path} does not exist."
            if not p.is_file():
                return f"Error: {path} is not a file."
            code = p.read_text()
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"
    
    if not code:
        return "Error: No code to analyze."
    
    # Run analyses
    all_issues = []
    
    # Syntax check
    syntax_issues = _analyze_syntax(code)
    all_issues.extend(syntax_issues)
    
    # If there are syntax errors, stop here
    if any(i["severity"] == "error" for i in syntax_issues):
        return _format_results(all_issues, path or "<provided code>")
    
    # Parse AST for further analysis
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _format_results(all_issues, path or "<provided code>")
    
    # Run AST-based analyses
    all_issues.extend(_analyze_imports(tree))
    all_issues.extend(_analyze_functions(tree))
    all_issues.extend(_analyze_complexity(tree))
    all_issues.extend(_analyze_patterns(code))
    
    return _format_results(all_issues, path or "<provided code>")


def _format_results(issues: list[dict], source: str) -> str:
    """Format analysis results."""
    if not issues:
        return f"✓ No issues found in {source}"
    
    # Group by severity
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    infos = [i for i in issues if i["severity"] == "info"]
    
    lines = [f"Analysis results for {source}:", ""]
    
    if errors:
        lines.append(f"Errors ({len(errors)}):")
        for issue in errors:
            lines.append(f"  Line {issue['line']}: {issue['message']}")
        lines.append("")
    
    if warnings:
        lines.append(f"Warnings ({len(warnings)}):")
        for issue in warnings:
            lines.append(f"  Line {issue['line']}: {issue['message']}")
        lines.append("")
    
    if infos:
        lines.append(f"Suggestions ({len(infos)}):")
        for issue in infos:
            lines.append(f"  Line {issue['line']}: {issue['message']}")
        lines.append("")
    
    lines.append(f"Total: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} suggestions")
    
    return "\n".join(lines)
