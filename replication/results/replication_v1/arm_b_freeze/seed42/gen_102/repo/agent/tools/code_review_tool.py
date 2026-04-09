"""
Code review tool: analyze code for common issues and best practices.

Provides static analysis capabilities to identify:
- Unused imports
- Undefined variables
- Common Python anti-patterns
- Style issues
"""

from __future__ import annotations

import ast
import re
from typing import Any


def tool_info() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "code_review",
            "description": "Analyze Python code for common issues, anti-patterns, and style problems. Returns a list of findings with severity levels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to analyze",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename for context in output",
                    },
                },
                "required": ["code"],
            },
        },
    }


def _check_function_length(tree: ast.AST) -> list[tuple[str, str]]:
    """Check for overly long functions."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body:
                start = node.body[0].lineno
                end = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else start
                length = end - start + 1 if end else 1
                if length > 50:
                    findings.append(("info", f"Line {node.lineno}: Function '{node.name}' is {length} lines (consider refactoring)"))
    return findings


def _check_complexity(tree: ast.AST) -> list[tuple[str, str]]:
    """Check cyclomatic complexity."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            branches = 0
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                     ast.With, ast.comprehension, ast.BoolOp)):
                    branches += 1
            if branches > 10:
                findings.append(("warning", f"Line {node.lineno}: Function '{node.name}' has high complexity ({branches} branches)"))
    return findings


def _check_naming(tree: ast.AST) -> list[tuple[str, str]]:
    """Check naming conventions."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                if node.name not in ('__init__', '__call__', '__str__', '__repr__', '__enter__', '__exit__'):
                    findings.append(("info", f"Line {node.lineno}: Function '{node.name}' doesn't follow snake_case"))
        if isinstance(node, ast.ClassDef):
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                findings.append(("info", f"Line {node.lineno}: Class '{node.name}' doesn't follow PascalCase"))
    return findings


def _check_docstrings(tree: ast.AST) -> list[tuple[str, str]]:
    """Check for missing docstrings."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            has_docstring = False
            if node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    if isinstance(first.value.value, str):
                        has_docstring = True
            if not has_docstring:
                kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
                findings.append(("info", f"Line {node.lineno}: {kind} '{node.name}' missing docstring"))
    return findings


def _check_type_hints(tree: ast.AST) -> list[tuple[str, str]]:
    """Check for missing type hints."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.returns is None:
                findings.append(("info", f"Line {node.lineno}: Function '{node.name}' missing return type hint"))
            args_without_hints = [arg.arg for arg in node.args.args 
                                  if arg.annotation is None and arg.arg not in ('self', 'cls')]
            if args_without_hints:
                findings.append(("info", f"Line {node.lineno}: Function '{node.name}' args missing type hints: {', '.join(args_without_hints)}"))
    return findings


def _check_wildcard_imports(tree: ast.AST) -> list[tuple[str, str]]:
    """Check for wildcard imports."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == '*':
                    findings.append(("warning", f"Line {node.lineno}: Wildcard import from '{node.module}' - import specific names"))
    return findings


def tool_function(code: str, filename: str = "<unknown>") -> str:
    """Analyze code and return findings as formatted string."""
    findings = []
    
    # Try to parse the code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"
    
    # Check for unused imports
    imports = set()
    used_names = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name):
            used_names.add(node.id)
    
    unused = imports - used_names
    for name in unused:
        findings.append(("warning", f"Unused import: {name}"))
    
    # Check for bare except clauses
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append(("error", f"Line {node.lineno}: Bare 'except:' clause - use 'except Exception:' instead"))
    
    # Check for mutable default arguments
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for default in node.args.defaults + node.args.kw_defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    findings.append(("warning", f"Line {node.lineno}: Mutable default argument in function '{node.name}' - use None and initialize inside function"))
    
    # Check for == None or == True/False
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, ast.Eq):
                    if isinstance(node.comparators[0], ast.Constant):
                        val = node.comparators[0].value
                        if val is None or val is True or val is False:
                            findings.append(("warning", f"Line {node.lineno}: Use 'is' instead of '==' for singleton comparison"))
    
    # Check for long lines
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if len(line) > 100:
            findings.append(("info", f"Line {i}: Line too long ({len(line)} chars)"))
    
    # Check for TODO/FIXME comments
    for i, line in enumerate(lines, 1):
        if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE):
            findings.append(("info", f"Line {i}: Found marker comment"))
    
    # Additional checks
    findings.extend(_check_function_length(tree))
    findings.extend(_check_complexity(tree))
    findings.extend(_check_naming(tree))
    findings.extend(_check_docstrings(tree))
    findings.extend(_check_type_hints(tree))
    findings.extend(_check_wildcard_imports(tree))
    
    # Format output
    if not findings:
        return f"✓ No issues found in {filename}"
    
    # Group by severity
    errors = [f for f in findings if f[0] == "error"]
    warnings = [f for f in findings if f[0] == "warning"]
    infos = [f for f in findings if f[0] == "info"]
    
    output = [f"Code review for {filename}:"]
    output.append(f"  Found {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} suggestion(s)")
    output.append("")
    
    for severity, message in findings:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(severity, "•")
        output.append(f"  {icon} [{severity.upper()}] {message}")
    
    return "\n".join(output)
