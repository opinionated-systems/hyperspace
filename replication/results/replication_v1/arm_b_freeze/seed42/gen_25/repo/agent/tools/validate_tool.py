"""
Validation tool: check Python syntax and basic code quality.

Provides tools to validate code before applying changes to prevent
breaking the codebase with syntax errors.
"""

from __future__ import annotations

import ast
import io
import logging
import sys
import traceback
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "validate_python",
        "description": "Validate Python code for syntax errors and basic issues. Checks: (1) syntax validity, (2) basic AST structure, (3) common pitfalls. Returns validation report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to validate",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional file path for context in error messages",
                },
            },
            "required": ["code"],
        },
    }


def _check_syntax(code: str, file_path: str = "<string>") -> tuple[bool, list[str]]:
    """Check Python syntax. Returns (is_valid, list_of_errors)."""
    errors = []
    try:
        ast.parse(code)
        return True, errors
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}")
        return False, errors
    except Exception as e:
        errors.append(f"Parse error: {e}")
        return False, errors


def _check_common_issues(code: str) -> list[str]:
    """Check for common code quality issues."""
    issues = []
    
    # Check for obvious indentation issues
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped and not line.startswith('#'):
            # Check for mixed tabs and spaces
            if '\t' in line and ' ' in line[:len(line) - len(stripped)]:
                issues.append(f"Line {i}: Mixed tabs and spaces in indentation")
    
    # Check for unclosed brackets (basic check)
    brackets = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for i, line in enumerate(lines, 1):
        for char in line:
            if char in brackets:
                stack.append((char, i))
            elif char in brackets.values():
                if stack and brackets[stack[-1][0]] == char:
                    stack.pop()
                else:
                    issues.append(f"Line {i}: Unmatched closing bracket '{char}'")
    
    for char, line_num in stack:
        issues.append(f"Line {line_num}: Unclosed bracket '{char}'")
    
    return issues


def _check_imports(code: str) -> list[str]:
    """Check for potentially problematic imports."""
    issues = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ('sys', 'os', 'subprocess', 'shutil'):
                        issues.append(f"Uses {alias.name} module - ensure safe usage")
            elif isinstance(node, ast.ImportFrom):
                if node.module in ('sys', 'os', 'subprocess', 'shutil'):
                    issues.append(f"Uses {node.module} module - ensure safe usage")
    except:
        pass  # Syntax errors already caught
    
    return issues


def tool_function(code: str, file_path: str = "<string>") -> str:
    """Validate Python code and return a report."""
    report = [f"=== Validation Report for {file_path} ==="]
    
    # Syntax check
    is_valid, syntax_errors = _check_syntax(code, file_path)
    if not is_valid:
        report.append("\n❌ SYNTAX ERRORS:")
        for err in syntax_errors:
            report.append(f"  - {err}")
        report.append("\n⚠️  Code has syntax errors and should not be applied!")
        return '\n'.join(report)
    else:
        report.append("\n✅ Syntax: Valid")
    
    # Common issues
    common_issues = _check_common_issues(code)
    if common_issues:
        report.append("\n⚠️  Common Issues:")
        for issue in common_issues:
            report.append(f"  - {issue}")
    else:
        report.append("\n✅ Common Issues: None found")
    
    # Import warnings
    import_warnings = _check_imports(code)
    if import_warnings:
        report.append("\n⚠️  Import Warnings:")
        for warning in import_warnings:
            report.append(f"  - {warning}")
    
    # Overall assessment
    total_issues = len(common_issues) + len(import_warnings)
    if total_issues == 0:
        report.append("\n✅ Overall: Code looks good!")
    else:
        report.append(f"\n⚠️  Overall: {total_issues} issue(s) found (non-critical)")
    
    return '\n'.join(report)
