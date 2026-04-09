"""
Code analysis tool: analyze Python code for quality issues.

Provides static analysis capabilities to identify common code quality issues
like unused imports, missing docstrings, long functions, and style violations.
Useful for improving code quality during self-improvement iterations.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code for quality issues. "
            "Detects unused imports, missing docstrings, long functions, "
            "and other common code quality problems. "
            "Returns a report with findings and recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the Python file to analyze.",
                },
                "check_docstrings": {
                    "type": "boolean",
                    "description": "Check for missing docstrings (default: True).",
                },
                "check_imports": {
                    "type": "boolean",
                    "description": "Check for unused imports (default: True).",
                },
                "max_function_length": {
                    "type": "integer",
                    "description": "Maximum lines for a function (default: 50).",
                },
            },
            "required": ["file_path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set allowed root directory for file access."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_path_allowed(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def _get_function_length(node: ast.FunctionDef) -> int:
    """Calculate the number of lines in a function."""
    if hasattr(node, 'end_lineno') and node.end_lineno is not None:
        return node.end_lineno - node.lineno
    return len(node.body)


def _analyze_file(
    file_path: str,
    check_docstrings: bool = True,
    check_imports: bool = True,
    max_function_length: int = 50,
) -> dict[str, Any]:
    """Analyze a Python file for code quality issues."""
    findings = {
        "unused_imports": [],
        "missing_docstrings": [],
        "long_functions": [],
        "complex_functions": [],
        "style_issues": [],
    }
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.splitlines()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error in file: {e}"}
    
    # Track imports and their usage
    imports: dict[str, tuple[str, int]] = {}  # name -> (import_line, lineno)
    imported_names: set[str] = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = (f"import {alias.name}", node.lineno)
                imported_names.add(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = (f"from {module} import {alias.name}", node.lineno)
                imported_names.add(name)
    
    # Find all name usages
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            # Handle module.attribute usage
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
    
    # Check for unused imports
    if check_imports:
        for name, (import_line, lineno) in imports.items():
            if name not in used_names and not name.startswith("_"):
                findings["unused_imports"].append({
                    "line": lineno,
                    "import": import_line,
                })
    
    # Check functions for docstrings and length
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            
            # Check docstring
            if check_docstrings and not ast.get_docstring(node):
                findings["missing_docstrings"].append({
                    "line": node.lineno,
                    "function": func_name,
                    "type": "function",
                })
            
            # Check function length
            func_length = _get_function_length(node)
            if func_length > max_function_length:
                findings["long_functions"].append({
                    "line": node.lineno,
                    "function": func_name,
                    "lines": func_length,
                })
            
            # Check cyclomatic complexity (simple version)
            complexity = 1
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
            
            if complexity > 10:
                findings["complex_functions"].append({
                    "line": node.lineno,
                    "function": func_name,
                    "complexity": complexity,
                })
        
        # Check classes for docstrings
        elif isinstance(node, ast.ClassDef) and check_docstrings:
            if not ast.get_docstring(node):
                findings["missing_docstrings"].append({
                    "line": node.lineno,
                    "class": node.name,
                    "type": "class",
                })
    
    # Check for style issues
    for i, line in enumerate(lines, 1):
        # Trailing whitespace
        if line.rstrip() != line:
            findings["style_issues"].append({
                "line": i,
                "issue": "trailing_whitespace",
                "message": "Line has trailing whitespace",
            })
        
        # Lines too long (> 100 chars)
        if len(line) > 100:
            findings["style_issues"].append({
                "line": i,
                "issue": "line_too_long",
                "message": f"Line is {len(line)} characters (max 100 recommended)",
            })
        
        # TODO/FIXME comments
        if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE):
            match = re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE)
            findings["style_issues"].append({
                "line": i,
                "issue": "todo_comment",
                "message": f"Found {match.group(1).upper()} comment",
            })
    
    return findings


def tool_function(
    file_path: str,
    check_docstrings: bool = True,
    check_imports: bool = True,
    max_function_length: int = 50,
) -> str:
    """Analyze Python code for quality issues.
    
    Args:
        file_path: Path to the Python file to analyze
        check_docstrings: Whether to check for missing docstrings
        check_imports: Whether to check for unused imports
        max_function_length: Maximum recommended function length in lines
        
    Returns:
        Formatted analysis report with findings and recommendations
    """
    if not file_path or not file_path.strip():
        return "Error: file_path cannot be empty"
    
    if not _is_path_allowed(file_path):
        return f"Error: Access to path '{file_path}' is not allowed"
    
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    
    if not file_path.endswith('.py'):
        return f"Error: File must be a Python file (.py): {file_path}"
    
    try:
        findings = _analyze_file(
            file_path,
            check_docstrings=check_docstrings,
            check_imports=check_imports,
            max_function_length=max_function_length,
        )
        
        if "error" in findings:
            return f"Analysis failed: {findings['error']}"
        
        # Count total issues
        total_issues = (
            len(findings["unused_imports"]) +
            len(findings["missing_docstrings"]) +
            len(findings["long_functions"]) +
            len(findings["complex_functions"]) +
            len(findings["style_issues"])
        )
        
        # Build report
        lines = [f"Code Analysis Report for: {file_path}", "=" * 50, ""]
        lines.append(f"Total issues found: {total_issues}")
        lines.append("")
        
        if total_issues == 0:
            lines.append("✓ No code quality issues detected!")
            return "\n".join(lines)
        
        # Unused imports
        if findings["unused_imports"]:
            lines.append("UNUSED IMPORTS:")
            lines.append("-" * 30)
            for item in findings["unused_imports"]:
                lines.append(f"  Line {item['line']}: {item['import']}")
            lines.append("")
        
        # Missing docstrings
        if findings["missing_docstrings"]:
            lines.append("MISSING DOCSTRINGS:")
            lines.append("-" * 30)
            for item in findings["missing_docstrings"]:
                if item.get("type") == "function":
                    lines.append(f"  Line {item['line']}: Function '{item['function']}'")
                else:
                    lines.append(f"  Line {item['line']}: Class '{item['class']}'")
            lines.append("")
        
        # Long functions
        if findings["long_functions"]:
            lines.append("LONG FUNCTIONS:")
            lines.append("-" * 30)
            for item in findings["long_functions"]:
                lines.append(f"  Line {item['line']}: '{item['function']}' ({item['lines']} lines)")
            lines.append("")
        
        # Complex functions
        if findings["complex_functions"]:
            lines.append("COMPLEX FUNCTIONS (high cyclomatic complexity):")
            lines.append("-" * 30)
            for item in findings["complex_functions"]:
                lines.append(f"  Line {item['line']}: '{item['function']}' (complexity: {item['complexity']})")
            lines.append("")
        
        # Style issues
        if findings["style_issues"]:
            lines.append("STYLE ISSUES:")
            lines.append("-" * 30)
            for item in findings["style_issues"]:
                lines.append(f"  Line {item['line']}: {item['message']}")
            lines.append("")
        
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 30)
        if findings["unused_imports"]:
            lines.append("• Remove unused imports to clean up the code")
        if findings["missing_docstrings"]:
            lines.append("• Add docstrings to functions and classes for better documentation")
        if findings["long_functions"]:
            lines.append("• Consider breaking long functions into smaller, focused functions")
        if findings["complex_functions"]:
            lines.append("• Simplify complex functions by extracting logic or reducing branches")
        if findings["style_issues"]:
            lines.append("• Fix style issues for consistent code formatting")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error analyzing file: {e}"
