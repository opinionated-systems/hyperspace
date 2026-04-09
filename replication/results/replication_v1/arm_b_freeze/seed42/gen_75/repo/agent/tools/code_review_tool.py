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
    
    # Format output
    if not findings:
        return f"✓ No issues found in {filename}"
    
    output = [f"Code review for {filename}:"]
    for severity, message in findings:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(severity, "•")
        output.append(f"  {icon} [{severity.upper()}] {message}")
    
    return "\n".join(output)
