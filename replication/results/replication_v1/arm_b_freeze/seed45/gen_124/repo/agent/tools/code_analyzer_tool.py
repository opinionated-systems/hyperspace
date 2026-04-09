"""
Code analyzer tool: analyze Python code for common issues.

Provides static analysis capabilities to detect:
- Syntax errors
- Unused imports
- Undefined variables
- Common code smells
- Style violations

This helps the meta agent identify areas for improvement in the codebase.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_analyzer",
        "description": (
            "Analyze Python code for common issues like syntax errors, "
            "unused imports, undefined variables, and style violations. "
            "Helps identify areas for improvement in the codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file or directory to analyze.",
                },
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["syntax", "imports", "undefined", "complexity", "style", "all"]
                    },
                    "description": "Types of checks to run (default: ['all']).",
                },
                "max_issues": {
                    "type": "integer",
                    "description": "Maximum number of issues to report per file (default: 20).",
                },
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict code analysis operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Analysis restricted to {_ALLOWED_ROOT}"
    return True, ""


def tool_function(
    path: str,
    checks: list[str] | None = None,
    max_issues: int = 20,
) -> str:
    """Analyze Python code for common issues."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if checks is None:
            checks = ["all"]
        
        run_all = "all" in checks
        
        # Collect files to analyze
        files_to_analyze = []
        if p.is_file():
            if p.suffix == ".py":
                files_to_analyze.append(p)
            else:
                return f"Error: {p} is not a Python file."
        else:
            files_to_analyze = list(p.rglob("*.py"))
            # Exclude __pycache__ directories
            files_to_analyze = [f for f in files_to_analyze if "__pycache__" not in str(f)]
        
        if not files_to_analyze:
            return f"No Python files found in {p}"
        
        # Analyze each file
        all_results = []
        total_issues = 0
        
        for file_path in files_to_analyze:
            issues = _analyze_file(
                file_path,
                run_syntax=run_all or "syntax" in checks,
                run_imports=run_all or "imports" in checks,
                run_undefined=run_all or "undefined" in checks,
                run_complexity=run_all or "complexity" in checks,
                run_style=run_all or "style" in checks,
                max_issues=max_issues,
            )
            if issues:
                all_results.append((file_path, issues))
                total_issues += len(issues)
        
        # Format results
        if not all_results:
            return f"✓ No issues found in {len(files_to_analyze)} file(s) analyzed."
        
        output = [f"Code Analysis Results: {total_issues} issue(s) found in {len(files_to_analyze)} file(s) analyzed.\n"]
        
        for file_path, issues in all_results:
            rel_path = file_path.relative_to(p) if p.is_dir() else file_path.name
            output.append(f"\n{rel_path}:")
            for issue in issues[:max_issues]:
                output.append(f"  Line {issue['line']}: [{issue['type']}] {issue['message']}")
            if len(issues) > max_issues:
                output.append(f"  ... and {len(issues) - max_issues} more issues")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error: {e}"


def _analyze_file(
    file_path: Path,
    run_syntax: bool,
    run_imports: bool,
    run_undefined: bool,
    run_complexity: bool,
    run_style: bool,
    max_issues: int,
) -> list[dict]:
    """Analyze a single Python file."""
    issues = []
    
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return [{"line": 0, "type": "ERROR", "message": f"Could not read file: {e}"}]
    
    # Syntax check
    if run_syntax:
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            issues.append({
                "line": e.lineno or 1,
                "type": "SYNTAX",
                "message": f"Syntax error: {e.msg}"
            })
            return issues  # Can't analyze further if syntax is broken
    else:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues  # Skip other checks if syntax is broken
    
    # Collect imports and names for analysis
    imports = set()
    imported_names = {}  # name -> line
    defined_names = set()
    used_names = set()
    function_params = set()  # Track function parameters
    
    for node in ast.walk(tree):
        # Track imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                name = alias.asname or alias.name.split(".")[0]
                imported_names[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                imports.add(full_name)
                name = alias.asname or alias.name
                imported_names[name] = node.lineno
        
        # Track defined names (functions, classes, variables)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_names.add(node.name)
            # Track function parameters (including self/cls)
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                function_params.add(arg.arg)
            if node.args.vararg:
                function_params.add(node.args.vararg.arg)
            if node.args.kwarg:
                function_params.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
        elif isinstance(node, ast.Lambda):
            # Track lambda parameters
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                function_params.add(arg.arg)
            if node.args.vararg:
                function_params.add(node.args.vararg.arg)
            if node.args.kwarg:
                function_params.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined_names.add(node.id)
        
        # Track used names
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
    
    # Check for unused imports
    if run_imports:
        for name, line in imported_names.items():
            if name not in used_names and not name.startswith("_"):
                issues.append({
                    "line": line,
                    "type": "IMPORT",
                    "message": f"Potentially unused import: '{name}'"
                })
    
    # Check for undefined names (simple heuristic)
    if run_undefined:
        builtin_names = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__.__class__))
        common_builtins = {
            "True", "False", "None", "len", "range", "enumerate", "zip",
            "map", "filter", "sum", "min", "max", "abs", "int", "str",
            "float", "list", "dict", "set", "tuple", "print", "open",
            "isinstance", "hasattr", "getattr", "super", "type", "object",
            "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
            "AttributeError", "RuntimeError", "ImportError", "ModuleNotFoundError",
            "json", "re", "os", "sys", "pathlib", "typing", "datetime",
            "itertools", "collections", "functools", "math", "random",
            "subprocess", "threading", "time", "logging", "hashlib",
            "annotations",  # __future__ import
            # Type annotations
            "bool", "int", "str", "float", "list", "dict", "set", "tuple",
            "Any", "Callable", "Optional", "Union", "Dict", "List", "Set",
            "Tuple", "Type", "Generator", "Iterator", "Iterable", "Sequence",
            # Exception types
            "FileNotFoundError", "PermissionError", "OSError", "IOError",
            "NotImplementedError", "StopIteration", "GeneratorExit",
            "SyntaxError", "NameError", "ZeroDivisionError",
            # Built-in functions
            "reversed", "sorted", "any", "all", "hex", "oct", "bin",
            "chr", "ord", "pow", "round", "divmod", "format", "repr",
            "vars", "locals", "globals", "dir", "help", "id", "hash",
            "callable", "staticmethod", "classmethod", "property",
            "next", "iter", "input", "exit", "quit",
        }
        
        # Special names that are always defined in certain contexts
        special_names = {"self", "cls", "args", "kwargs", "e", "e2", "e3", "ex", "exc", "err", "error"}  # Common exception variable names
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if (node.id not in defined_names and 
                    node.id not in imported_names and
                    node.id not in builtin_names and
                    node.id not in common_builtins and
                    node.id not in function_params and
                    node.id not in special_names and
                    not node.id.startswith("__")):
                    issues.append({
                        "line": node.lineno,
                        "type": "UNDEFINED",
                        "message": f"Potentially undefined name: '{node.id}'"
                    })
    
    # Check complexity
    if run_complexity:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Simple complexity: count branches
                branches = 0
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, 
                                         ast.With, ast.Try, ast.ExceptHandler)):
                        branches += 1
                    elif isinstance(child, ast.BoolOp):
                        branches += len(child.values) - 1
                
                if branches > 10:
                    issues.append({
                        "line": node.lineno,
                        "type": "COMPLEXITY",
                        "message": f"Function '{node.name}' has high complexity ({branches} branches)"
                    })
    
    # Style checks
    if run_style:
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            # Line too long
            if len(line) > 120:
                issues.append({
                    "line": i,
                    "type": "STYLE",
                    "message": f"Line too long ({len(line)} > 120 characters)"
                })
            
            # Trailing whitespace
            if line.rstrip() != line:
                issues.append({
                    "line": i,
                    "type": "STYLE",
                    "message": "Trailing whitespace"
                })
            
            # Mixed tabs and spaces
            if "\t" in line and " " in line:
                stripped = line.lstrip()
                if stripped and not line.startswith(stripped):
                    indent = line[:len(line) - len(stripped)]
                    if "\t" in indent and " " in indent:
                        issues.append({
                            "line": i,
                            "type": "STYLE",
                            "message": "Mixed tabs and spaces in indentation"
                        })
        
        # Check for TODO/FIXME comments
        todo_pattern = re.compile(r'#.*\b(TODO|FIXME|XXX|HACK)\b', re.IGNORECASE)
        for i, line in enumerate(lines, 1):
            if todo_pattern.search(line):
                match = todo_pattern.search(line)
                issues.append({
                    "line": i,
                    "type": "NOTE",
                    "message": f"Found {match.group(1).upper()} comment"
                })
    
    return issues[:max_issues]
