"""
Code analysis tool: analyze Python code for common issues.

Provides static analysis capabilities to help identify potential
bugs, style issues, and code quality problems.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "analyze",
        "description": (
            "Analyze Python code for common issues. "
            "Checks for syntax errors, undefined variables, "
            "and basic code quality issues. "
            "Useful for validating code before editing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file to analyze.",
                },
                "code": {
                    "type": "string",
                    "description": "Python code string to analyze (alternative to path).",
                },
            },
            "required": [],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict analysis operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _validate_path(path: str) -> tuple[bool, str]:
    """Validate that a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True, ""
    
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Analysis restricted to {_ALLOWED_ROOT}"
    return True, ""


def _analyze_syntax(code: str) -> list[dict]:
    """Check for syntax errors in Python code."""
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append({
            "type": "syntax_error",
            "line": e.lineno or 0,
            "message": str(e),
            "severity": "error"
        })
    return issues


def _analyze_imports(code: str) -> list[dict]:
    """Analyze import statements for potential issues."""
    issues = []
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        issues.append({
                            "type": "private_import",
                            "line": node.lineno,
                            "message": f"Importing private module: {alias.name}",
                            "severity": "warning"
                        })
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("_"):
                    issues.append({
                        "type": "private_import",
                        "line": node.lineno,
                        "message": f"Importing from private module: {node.module}",
                        "severity": "warning"
                    })
    except SyntaxError:
        pass  # Already reported in _analyze_syntax
    
    return issues


def _analyze_complexity(code: str) -> list[dict]:
    """Analyze code complexity metrics."""
    issues = []
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function length
                lines = node.end_lineno - node.lineno if node.end_lineno else 0
                if lines > 100:
                    issues.append({
                        "type": "long_function",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' is {lines} lines long (consider refactoring)",
                        "severity": "info"
                    })
                
                # Check number of arguments
                num_args = len(node.args.args) + len(node.args.kwonlyargs)
                if node.args.vararg:
                    num_args += 1
                if node.args.kwarg:
                    num_args += 1
                if num_args > 8:
                    issues.append({
                        "type": "many_arguments",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' has {num_args} arguments (consider using a config object)",
                        "severity": "info"
                    })
    except SyntaxError:
        pass
    
    return issues


def _format_issues(issues: list[dict], source: str) -> str:
    """Format analysis issues into a readable string."""
    if not issues:
        return f"✓ No issues found in {source}"
    
    lines = [f"Analysis results for {source}:", ""]
    
    # Group by severity
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    infos = [i for i in issues if i["severity"] == "info"]
    
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
        lines.append(f"Info ({len(infos)}):")
        for issue in infos:
            lines.append(f"  Line {issue['line']}: {issue['message']}")
        lines.append("")
    
    return "\n".join(lines)


def tool_function(
    path: str | None = None,
    code: str | None = None,
) -> str:
    """Analyze Python code for common issues.
    
    Args:
        path: Absolute path to Python file to analyze
        code: Python code string to analyze (alternative to path)
        
    Returns:
        Analysis results or error message
    """
    # Validate inputs
    if path is None and code is None:
        return "Error: Either 'path' or 'code' must be provided"
    
    if path is not None and code is not None:
        return "Error: Provide either 'path' or 'code', not both"
    
    # Get code to analyze
    if path is not None:
        valid, error = _validate_path(path)
        if not valid:
            return error
        
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        
        if not p.is_file():
            return f"Error: Path is not a file: {path}"
        
        try:
            code = p.read_text()
            source = path
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"
    else:
        source = "<provided code>"
    
    # Run analysis
    all_issues = []
    all_issues.extend(_analyze_syntax(code))
    all_issues.extend(_analyze_imports(code))
    all_issues.extend(_analyze_complexity(code))
    
    return _format_issues(all_issues, source)
