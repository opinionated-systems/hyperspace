"""
Code analysis tool: analyze Python code for common issues.

Provides static analysis capabilities to help identify potential bugs,
style issues, and code quality problems.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code for common issues and quality metrics. "
            "Commands: analyze_file (check a single file), "
            "analyze_directory (check all Python files in a directory), "
            "check_syntax (validate Python syntax)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["analyze_file", "analyze_directory", "check_syntax"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory to analyze.",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to analyze (default: 50).",
                },
            },
            "required": ["command", "path"],
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
            return False, f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def tool_function(
    command: str,
    path: str,
    max_files: int = 50,
) -> str:
    """Execute a code analysis command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if command == "analyze_file":
            return _analyze_file(p)
        elif command == "analyze_directory":
            return _analyze_directory(p, max_files)
        elif command == "check_syntax":
            return _check_syntax(p)
        else:
            return f"Error: unknown command '{command}'"
    except Exception as e:
        return f"Error: {e}"


def _analyze_file(path: Path) -> str:
    """Analyze a single Python file."""
    if not path.is_file():
        return f"Error: {path} is not a file"
    
    if path.suffix != ".py":
        return f"Error: {path} is not a Python file (.py)"
    
    try:
        content = path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"
    
    issues = []
    metrics = {}
    
    # Check syntax
    try:
        tree = ast.parse(content)
        metrics["syntax_valid"] = True
    except SyntaxError as e:
        issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
        metrics["syntax_valid"] = False
        return f"Analysis of {path}:\n\nSyntax Error:\n" + "\n".join(issues)
    
    # Analyze AST
    metrics["total_lines"] = len(content.splitlines())
    metrics["blank_lines"] = len([l for l in content.splitlines() if not l.strip()])
    metrics["code_lines"] = metrics["total_lines"] - metrics["blank_lines"]
    
    # Count various constructs
    function_count = 0
    class_count = 0
    import_count = 0
    docstring_count = 0
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_count += 1
            if ast.get_docstring(node):
                docstring_count += 1
        elif isinstance(node, ast.ClassDef):
            class_count += 1
            if ast.get_docstring(node):
                docstring_count += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            import_count += 1
    
    metrics["functions"] = function_count
    metrics["classes"] = class_count
    metrics["imports"] = import_count
    metrics["docstrings"] = docstring_count
    
    # Check for common issues
    lines = content.splitlines()
    
    # Check for bare except clauses
    for i, line in enumerate(lines, 1):
        if re.search(r'except\s*:', line) and 'except Exception' not in line:
            issues.append(f"Line {i}: Bare 'except:' clause (catches SystemExit, KeyboardInterrupt)")
    
    # Check for mutable default arguments
    for i, line in enumerate(lines, 1):
        if re.search(r'def\s+\w+\s*\([^)]*=\s*(\[|\{)', line):
            issues.append(f"Line {i}: Mutable default argument (list or dict)")
    
    # Check for unused imports (simple heuristic)
    imported_names = set()
    used_names = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Name):
            used_names.add(node.id)
    
    # Check for print statements (potential debug code)
    for i, line in enumerate(lines, 1):
        if re.search(r'\bprint\s*\(', line) and not line.strip().startswith('#'):
            issues.append(f"Line {i}: Print statement (potential debug code)")
    
    # Build report
    report = [f"Analysis of {path}:", ""]
    report.append("Metrics:")
    report.append(f"  Total lines: {metrics['total_lines']}")
    report.append(f"  Code lines: {metrics['code_lines']}")
    report.append(f"  Functions: {metrics['functions']}")
    report.append(f"  Classes: {metrics['classes']}")
    report.append(f"  Imports: {metrics['imports']}")
    report.append(f"  Docstrings: {metrics['docstrings']}/{metrics['functions'] + metrics['classes']}")
    
    if issues:
        report.append("")
        report.append("Issues found:")
        for issue in issues[:20]:  # Limit issues
            report.append(f"  - {issue}")
        if len(issues) > 20:
            report.append(f"  ... and {len(issues) - 20} more issues")
    else:
        report.append("")
        report.append("No obvious issues found.")
    
    return "\n".join(report)


def _analyze_directory(path: Path, max_files: int) -> str:
    """Analyze all Python files in a directory."""
    if not path.is_dir():
        return f"Error: {path} is not a directory"
    
    py_files = list(path.rglob("*.py"))
    py_files = [f for f in py_files if not any(part.startswith("__pycache__") for part in f.parts)]
    
    if not py_files:
        return f"No Python files found in {path}"
    
    # Limit files
    if len(py_files) > max_files:
        py_files = py_files[:max_files]
        truncated = True
    else:
        truncated = False
    
    total_issues = 0
    total_lines = 0
    file_reports = []
    
    for py_file in py_files:
        try:
            result = _analyze_file(py_file)
            # Extract issue count
            if "Issues found:" in result:
                issue_count = result.count("Line ")
                total_issues += issue_count
            # Extract line count
            for line in result.split("\n"):
                if "Total lines:" in line:
                    try:
                        total_lines += int(line.split(":")[1].strip())
                    except:
                        pass
            file_reports.append(f"\n{'='*60}\n{result}")
        except Exception as e:
            file_reports.append(f"\n{'='*60}\nError analyzing {py_file}: {e}")
    
    summary = [
        f"Directory Analysis: {path}",
        f"Files analyzed: {len(py_files)}{' (truncated)' if truncated else ''}",
        f"Total lines: {total_lines}",
        f"Total issues: {total_issues}",
    ]
    
    if truncated:
        summary.append(f"Note: Only first {max_files} files analyzed")
    
    return "\n".join(summary + file_reports)


def _check_syntax(path: Path) -> str:
    """Check Python syntax of a file or directory."""
    if path.is_file():
        if path.suffix != ".py":
            return f"Error: {path} is not a Python file"
        
        try:
            content = path.read_text()
            ast.parse(content)
            return f"✓ {path}: Syntax OK"
        except SyntaxError as e:
            return f"✗ {path}: Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return f"✗ {path}: Error reading file: {e}"
    
    elif path.is_dir():
        py_files = list(path.rglob("*.py"))
        py_files = [f for f in py_files if not any(part.startswith("__pycache__") for part in f.parts)]
        
        results = []
        errors = 0
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                ast.parse(content)
                results.append(f"✓ {py_file}")
            except SyntaxError as e:
                results.append(f"✗ {py_file}: Line {e.lineno}: {e.msg}")
                errors += 1
            except Exception as e:
                results.append(f"✗ {py_file}: {e}")
                errors += 1
        
        summary = f"Syntax check: {len(py_files)} files, {errors} errors"
        return summary + "\n" + "\n".join(results)
    
    else:
        return f"Error: {path} is not a file or directory"
