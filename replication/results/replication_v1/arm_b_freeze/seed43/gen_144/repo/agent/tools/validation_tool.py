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
            "import tests, and basic linting. "
            "Helps verify that modifications don't break the codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["check_syntax", "check_imports", "run_linter", "test_function"],
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
    """Check Python syntax without executing."""
    if path:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    if not code:
        return "Error: No code provided"
    
    try:
        ast.parse(code)
        return "Syntax check passed: Code is valid Python."
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
    except Exception as e:
        return f"Parse error: {e}"


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


def tool_function(
    command: str,
    path: str | None = None,
    code: str | None = None,
    function_name: str | None = None,
    test_input: dict | None = None,
) -> str:
    """Execute a validation command."""
    valid_commands = ["check_syntax", "check_imports", "run_linter", "test_function"]
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
    
    return "Error: Unknown command"
