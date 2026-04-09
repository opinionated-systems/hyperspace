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
            "Analyze Python code for common issues. "
            "Commands: lint (check for syntax errors, unused imports, etc.), "
            "complexity (calculate cyclomatic complexity), "
            "imports (list all imports in a file)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["lint", "complexity", "imports"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Python file to analyze.",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str) -> str:
    """Execute a code analysis command."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        if not p.is_file():
            return f"Error: {path} is not a file."
        
        if not str(p).endswith('.py'):
            return f"Error: {path} is not a Python file."
        
        content = p.read_text()
        
        if command == "lint":
            return _lint(content, str(p))
        elif command == "complexity":
            return _complexity(content, str(p))
        elif command == "imports":
            return _imports(content, str(p))
        else:
            return f"Error: unknown command {command}"
    
    except Exception as e:
        return f"Error: {e}"


def _lint(content: str, path: str) -> str:
    """Check for common Python issues."""
    issues = []
    
    # Check for syntax errors
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Syntax error in {path}: {e}"
    
    # Check for unused imports
    imports = set()
    used_names = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name):
            used_names.add(node.id)
    
    unused = imports - used_names
    if unused:
        issues.append(f"Potentially unused imports: {', '.join(unused)}")
    
    # Check for bare except clauses
    bare_except_pattern = r'except\s*:'
    if re.search(bare_except_pattern, content):
        issues.append("Found bare 'except:' clause - consider using 'except Exception:'")
    
    # Check for TODO/FIXME comments
    todo_pattern = r'#\s*(TODO|FIXME|XXX|HACK)'
    todos = re.findall(todo_pattern, content, re.IGNORECASE)
    if todos:
        issues.append(f"Found {len(todos)} TODO/FIXME/XXX/HACK comments")
    
    # Check for print statements (potential debugging code)
    print_pattern = r'^\s*print\s*\('
    prints = re.findall(print_pattern, content, re.MULTILINE)
    if prints:
        issues.append(f"Found {len(prints)} print statements - consider using logging instead")
    
    # Check for long lines
    long_lines = []
    for i, line in enumerate(content.split('\n'), 1):
        if len(line) > 100:
            long_lines.append(i)
    if long_lines:
        issues.append(f"Lines exceeding 100 characters: {long_lines[:5]}{'...' if len(long_lines) > 5 else ''}")
    
    if issues:
        return f"Lint results for {path}:\n" + "\n".join(f"- {issue}" for issue in issues)
    else:
        return f"No major issues found in {path}"


def _complexity(content: str, path: str) -> str:
    """Calculate cyclomatic complexity of functions."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Syntax error in {path}: {e}"
    
    complexities = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = 1  # Base complexity
            
            # Count branches
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
                elif isinstance(child, ast.comprehension):
                    complexity += 1
            
            complexities.append((node.name, complexity))
    
    if complexities:
        result = f"Cyclomatic complexity for {path}:\n"
        for name, comp in sorted(complexities, key=lambda x: -x[1]):
            level = "low" if comp <= 5 else "medium" if comp <= 10 else "high"
            result += f"- {name}: {comp} ({level})\n"
        return result
    else:
        return f"No functions found in {path}"


def _imports(content: str, path: str) -> str:
    """List all imports in a file."""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Syntax error in {path}: {e}"
    
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imports.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in node.names)
            imports.append(f"from {module} import {names}")
    
    if imports:
        return f"Imports in {path}:\n" + "\n".join(f"- {imp}" for imp in imports)
    else:
        return f"No imports found in {path}"
