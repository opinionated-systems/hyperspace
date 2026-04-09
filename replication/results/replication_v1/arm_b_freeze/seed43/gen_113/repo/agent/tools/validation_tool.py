"""
Validation tool: test and verify code changes.

Provides utilities for testing Python code, running linters,
and validating file changes.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "validation",
        "description": (
            "Validate code changes by running syntax checks, "
            "import tests, basic linting, and complexity analysis. "
            "Helps verify that modifications don't break the codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["check_syntax", "check_imports", "run_linter", "test_function", "analyze_complexity"],
                    "description": "The validation command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file to validate.",
                },
                "code": {
                    "type": "string",
                    "description": "Python code string to validate (for check_syntax).",
                },
                "function_name": {
                    "type": "string",
                    "description": "Function name to test (for test_function).",
                },
                "test_input": {
                    "type": "object",
                    "description": "Input dict for function test (for test_function).",
                },
            },
            "required": ["command"],
        },
    }


def _check_syntax(code: str | None = None, path: str | None = None) -> str:
    """Check Python syntax without executing.
    
    Args:
        code: Python code string to check (optional if path provided)
        path: Path to Python file to check (optional if code provided)
        
    Returns:
        Syntax check result message
    """
    if path:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError:
            return f"Error: Permission denied reading file: {path}"
        except UnicodeDecodeError as e:
            return f"Error: File encoding issue: {e}"
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"
    
    if not code or not isinstance(code, str):
        return "Error: No code provided or code is not a string"
    
    # Check for empty or whitespace-only code
    if not code.strip():
        return "Error: Code is empty or whitespace only"
    
    try:
        tree = ast.parse(code)
        
        # Additional checks for common issues
        issues = []
        
        # Check for undefined variables (basic check)
        defined_names = set()
        undefined_names = set()
        imported_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
                # Add function arguments
                for arg in node.args.args:
                    defined_names.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    defined_names.add(arg.arg)
                if node.args.vararg:
                    defined_names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined_names.add(node.args.kwarg.arg)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_names.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    if node.id not in defined_names and node.id not in dir(__builtins__):
                        undefined_names.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                else:
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
        
        # Remove imported names from undefined
        undefined_names -= imported_names
        
        if undefined_names:
            issues.append(f"Potentially undefined names: {', '.join(sorted(undefined_names))}")
        
        # Check for common anti-patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append("Bare except clause found (catches all exceptions including KeyboardInterrupt)")
                elif isinstance(node.type, ast.Name) and node.type.id == 'Exception':
                    issues.append("Generic 'except Exception' found (consider catching specific exceptions)")
        
        if issues:
            return f"Syntax check passed, but warnings found:\n" + "\n".join(f"  - {issue}" for issue in issues)
        
        return "Syntax check passed: Code is valid Python."
        
    except SyntaxError as e:
        # Provide more helpful error message with context
        lines = code.split('\n')
        error_msg = f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
        if 1 <= e.lineno <= len(lines):
            error_line = lines[e.lineno - 1]
            error_msg += f"\n  Line {e.lineno}: {error_line}"
            if e.offset and e.offset <= len(error_line):
                pointer = ' ' * (e.offset - 1) + '^'
                error_msg += f"\n         {pointer}"
        return error_msg
    except Exception as e:
        return f"Parse error: {type(e).__name__}: {e}"


def _check_imports(path: str) -> str:
    """Try to import a module and report any errors."""
    try:
        # Add the parent directory to path for imports
        p = Path(path)
        if p.exists():
            parent = p.parent
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            
            # Try importing the module
            module_name = p.stem
            __import__(module_name)
            return f"Import check passed: '{module_name}' imported successfully."
        else:
            return f"Error: File not found: {path}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Import error: {e}\n{tb}"


def _run_linter(path: str) -> str:
    """Run basic linting checks on a Python file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        return f"Error reading file: {e}"
    
    issues = []
    lines = code.split('\n')
    
    # Check for common issues
    for i, line in enumerate(lines, 1):
        # Check for trailing whitespace
        if line.rstrip() != line:
            issues.append(f"Line {i}: Trailing whitespace")
        
        # Check for tabs (should use spaces)
        if '\t' in line:
            issues.append(f"Line {i}: Contains tabs, should use spaces")
        
        # Check for very long lines
        if len(line) > 120:
            issues.append(f"Line {i}: Line too long ({len(line)} chars)")
        
        # Check for bare except
        if re.match(r'^\s*except\s*:', line):
            issues.append(f"Line {i}: Bare 'except:' clause (should specify exception type)")
        
        # Check for print statements (might be debug code)
        if re.search(r'\bprint\s*\(', line) and '"debug"' not in line.lower():
            issues.append(f"Line {i}: Contains print statement (consider using logging)")
    
    # Check for common Python issues
    if 'import *' in code:
        issues.append("Uses 'import *' (should use explicit imports)")
    
    if issues:
        return "Linting issues found:\n" + "\n".join(f"  - {issue}" for issue in issues)
    else:
        return "Linter check passed: No obvious issues found."


def _test_function(path: str, function_name: str, test_input: dict | None = None) -> str:
    """Test a specific function in a module."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        
        # Add parent to path
        parent = p.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        
        # Import the module
        module_name = p.stem
        module = __import__(module_name)
        
        # Get the function
        if not hasattr(module, function_name):
            available = [name for name in dir(module) if not name.startswith('_')]
            return f"Error: Function '{function_name}' not found. Available: {available}"
        
        func = getattr(module, function_name)
        
        # Call the function
        if test_input is None:
            test_input = {}
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            result = func(**test_input)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            result_str = repr(result)
            if len(result_str) > 500:
                result_str = result_str[:250] + "..." + result_str[-250:]
            
            return f"Function test passed:\n  Result: {result_str}\n  Output: {output[:200] if output else '(none)'}"
        except Exception as e:
            sys.stdout = old_stdout
            tb = traceback.format_exc()
            return f"Function test failed: {e}\n{tb}"
            
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error testing function: {e}\n{tb}"


def _analyze_complexity(path: str) -> str:
    """Analyze code complexity metrics."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error in file: {e}"
        
        # Calculate metrics
        total_lines = len(code.split('\n'))
        non_empty_lines = len([l for l in code.split('\n') if l.strip()])
        
        functions = []
        classes = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate cyclomatic complexity (basic approximation)
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                
                functions.append({
                    'name': node.name,
                    'lines': node.end_lineno - node.lineno if node.end_lineno else 0,
                    'complexity': complexity,
                    'args': len(node.args.args),
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'lines': node.end_lineno - node.lineno if node.end_lineno else 0,
                    'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                else:
                    imports.append(f"{node.module}.{node.names[0].name}" if node.names else node.module)
        
        # Build report
        report = [f"Complexity Analysis for {path}:", "=" * 50]
        report.append(f"Total lines: {total_lines}")
        report.append(f"Non-empty lines: {non_empty_lines}")
        report.append(f"Classes: {len(classes)}")
        report.append(f"Functions: {len(functions)}")
        report.append(f"Imports: {len(imports)}")
        
        if classes:
            report.append("\nClasses:")
            for cls in classes:
                report.append(f"  - {cls['name']}: {cls['lines']} lines, {cls['methods']} methods")
        
        if functions:
            report.append("\nFunctions (sorted by complexity):")
            sorted_funcs = sorted(functions, key=lambda x: x['complexity'], reverse=True)
            for func in sorted_funcs[:10]:  # Top 10 most complex
                report.append(f"  - {func['name']}: {func['lines']} lines, complexity {func['complexity']}, {func['args']} args")
        
        # Complexity warnings
        high_complexity = [f for f in functions if f['complexity'] > 10]
        if high_complexity:
            report.append(f"\n⚠️  High complexity functions (>10): {len(high_complexity)}")
            for func in high_complexity:
                report.append(f"    - {func['name']} (complexity: {func['complexity']})")
        
        long_functions = [f for f in functions if f['lines'] > 50]
        if long_functions:
            report.append(f"\n⚠️  Long functions (>50 lines): {len(long_functions)}")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error analyzing complexity: {e}"


def tool_function(
    command: str,
    path: str | None = None,
    code: str | None = None,
    function_name: str | None = None,
    test_input: dict | None = None,
) -> str:
    """Execute a validation command."""
    valid_commands = ["check_syntax", "check_imports", "run_linter", "test_function", "analyze_complexity"]
    if command not in valid_commands:
        return f"Error: Unknown command '{command}'. Valid: {', '.join(valid_commands)}"
    
    if command == "check_syntax":
        return _check_syntax(code, path)
    elif command == "check_imports":
        if not path:
            return "Error: 'path' required for check_imports"
        return _check_imports(path)
    elif command == "run_linter":
        if not path:
            return "Error: 'path' required for run_linter"
        return _run_linter(path)
    elif command == "test_function":
        if not path or not function_name:
            return "Error: 'path' and 'function_name' required for test_function"
        return _test_function(path, function_name, test_input)
    elif command == "analyze_complexity":
        if not path:
            return "Error: 'path' required for analyze_complexity"
        return _analyze_complexity(path)
    
    return "Error: Unknown command"
