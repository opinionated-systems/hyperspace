"""
Code analysis tool: analyze Python code for common issues.

Provides static analysis capabilities to help identify potential bugs,
style issues, and code quality problems.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code for common issues like syntax errors, "
            "undefined variables, unused imports, and style problems. "
            "Provides actionable feedback for code improvement."
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
            "oneOf": [
                {"required": ["path"]},
                {"required": ["code"]},
            ],
        },
    }


def _analyze_code(source: str, filename: str = "<string>") -> dict:
    """Analyze Python source code and return findings."""
    issues = []
    warnings = []
    
    # Check for syntax errors first
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {
            "valid": False,
            "issues": [f"Syntax error at line {e.lineno}: {e.msg}"],
            "warnings": [],
            "metrics": {},
        }
    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Parse error: {e}"],
            "warnings": [],
            "metrics": {},
        }
    
    # Collect defined and used names
    defined_names = set()
    used_names = set()
    imported_names = {}
    
    for node in ast.walk(tree):
        # Function definitions
        if isinstance(node, ast.FunctionDef):
            defined_names.add(node.name)
            # Check for empty functions
            if not node.body or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                warnings.append(f"Function '{node.name}' is empty (line {node.lineno})")
            # Check for long functions
            func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
            if func_lines > 50:
                warnings.append(f"Function '{node.name}' is very long ({func_lines} lines, line {node.lineno})")
        
        # Class definitions
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
            # Check for empty classes
            if not node.body or all(isinstance(n, ast.Pass) for n in node.body):
                warnings.append(f"Class '{node.name}' is empty (line {node.lineno})")
        
        # Variable assignments
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        
        # Imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names[name] = alias.name
                defined_names.add(name)
        
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names[name] = f"{module}.{alias.name}" if module else alias.name
                defined_names.add(name)
        
        # Check for bare except clauses
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                warnings.append(f"Bare 'except:' clause at line {node.lineno} (should catch specific exceptions)")
    
    # Check for unused imports
    for name, full_name in imported_names.items():
        if name not in used_names and not name.startswith("_"):
            # Check if it's a module-level import used elsewhere
            if "." in full_name:
                base_module = full_name.split(".")[0]
                if base_module not in used_names:
                    warnings.append(f"Potentially unused import: '{name}'")
    
    # Check for undefined names (simple heuristic)
    builtin_names = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    undefined = used_names - defined_names - builtin_names
    # Filter out common false positives
    undefined = {name for name in undefined if not name.startswith("__")}
    for name in undefined:
        warnings.append(f"Potentially undefined name: '{name}'")
    
    # Calculate metrics
    lines = source.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]
    
    metrics = {
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty_lines),
        "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
        "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
        "imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
    }
    
    # Check for common style issues
    for i, line in enumerate(lines, 1):
        # Trailing whitespace
        if line.rstrip() != line:
            warnings.append(f"Line {i}: trailing whitespace")
        # Lines too long
        if len(line) > 120:
            warnings.append(f"Line {i}: line too long ({len(line)} characters)")
        # Mixed tabs and spaces
        if "\t" in line and " " in line:
            warnings.append(f"Line {i}: mixed tabs and spaces")
    
    return {
        "valid": True,
        "issues": issues,
        "warnings": warnings,
        "metrics": metrics,
    }


def tool_function(path: str | None = None, code: str | None = None) -> str:
    """Analyze Python code and return findings."""
    if path:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: File not found: {path}"
            if not p.is_file():
                return f"Error: Not a file: {path}"
            source = p.read_text()
            filename = str(p)
        except Exception as e:
            return f"Error reading file: {e}"
    elif code:
        source = code
        filename = "<string>"
    else:
        return "Error: Either 'path' or 'code' must be provided"
    
    result = _analyze_code(source, filename)
    
    # Format output
    lines = [f"Code Analysis: {filename}"]
    lines.append("=" * 50)
    
    if not result["valid"]:
        lines.append("\n❌ Code is INVALID:")
        for issue in result["issues"]:
            lines.append(f"  - {issue}")
        return "\n".join(lines)
    
    lines.append("\n✅ Code is syntactically valid")
    
    # Metrics
    lines.append("\n📊 Metrics:")
    for key, value in result["metrics"].items():
        lines.append(f"  - {key}: {value}")
    
    # Issues
    if result["issues"]:
        lines.append("\n❌ Issues:")
        for issue in result["issues"]:
            lines.append(f"  - {issue}")
    
    # Warnings
    if result["warnings"]:
        lines.append(f"\n⚠️ Warnings ({len(result['warnings'])}):")
        for warning in result["warnings"][:20]:  # Limit to 20 warnings
            lines.append(f"  - {warning}")
        if len(result["warnings"]) > 20:
            lines.append(f"  ... and {len(result['warnings']) - 20} more warnings")
    else:
        lines.append("\n✨ No warnings found!")
    
    return "\n".join(lines)
