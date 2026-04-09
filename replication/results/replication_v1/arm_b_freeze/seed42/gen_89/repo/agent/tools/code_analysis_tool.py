"""
Code analysis tool: analyze Python code for quality metrics and issues.

Provides static analysis capabilities to help the agent understand code structure,
identify potential issues, and suggest improvements.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code for quality metrics, structure, and potential issues. "
            "Provides insights on code complexity, documentation coverage, and common issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["analyze_file", "analyze_directory", "metrics", "find_issues"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory to analyze.",
                },
                "max_complexity": {
                    "type": "integer",
                    "description": "Maximum cyclomatic complexity threshold (default: 10).",
                    "default": 10,
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(
    command: str,
    path: str,
    max_complexity: int = 10,
) -> str:
    """Execute a code analysis command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if command == "analyze_file":
            return _analyze_file(p, max_complexity)
        elif command == "analyze_directory":
            return _analyze_directory(p, max_complexity)
        elif command == "metrics":
            return _get_metrics(p)
        elif command == "find_issues":
            return _find_issues(p)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _analyze_file(p: Path, max_complexity: int) -> str:
    """Analyze a single Python file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_file():
        return f"Error: {p} is not a file."
    
    if not str(p).endswith('.py'):
        return f"Error: {p} is not a Python file."

    try:
        content = p.read_text()
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error: Syntax error in {p}: {e}"
    except Exception as e:
        return f"Error reading {p}: {e}"

    # Calculate metrics
    lines = content.split('\n')
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    blank_lines = len([l for l in lines if not l.strip()])
    comment_lines = total_lines - code_lines - blank_lines

    # Count definitions
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

    # Calculate complexity
    complexities = []
    for func in functions:
        complexity = _calculate_complexity(func)
        complexities.append((func.name, complexity))

    # Check docstrings
    documented = 0
    for func in functions:
        if ast.get_docstring(func):
            documented += 1
    
    doc_coverage = (documented / len(functions) * 100) if functions else 100

    # Build report
    report = [
        f"Analysis of {p}:",
        "=" * 50,
        f"Lines: {total_lines} total, {code_lines} code, {comment_lines} comments, {blank_lines} blank",
        f"Functions: {len(functions)}, Classes: {len(classes)}, Imports: {len(imports)}",
        f"Documentation coverage: {doc_coverage:.1f}%",
        "",
        "Function Complexity:",
    ]

    for name, complexity in sorted(complexities, key=lambda x: x[1], reverse=True):
        status = "⚠️" if complexity > max_complexity else "✓"
        report.append(f"  {status} {name}: {complexity}")

    # High complexity warning
    high_complexity = [(n, c) for n, c in complexities if c > max_complexity]
    if high_complexity:
        report.extend([
            "",
            f"Warning: {len(high_complexity)} function(s) exceed complexity threshold ({max_complexity}):",
        ])
        for name, complexity in high_complexity:
            report.append(f"  - {name}: {complexity}")

    return "\n".join(report)


def _analyze_directory(p: Path, max_complexity: int) -> str:
    """Analyze all Python files in a directory."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_dir():
        return f"Error: {p} is not a directory."

    py_files = list(p.rglob("*.py"))
    if not py_files:
        return f"No Python files found in {p}"

    total_metrics = {
        "files": 0,
        "total_lines": 0,
        "code_lines": 0,
        "functions": 0,
        "classes": 0,
        "imports": 0,
        "documented": 0,
    }

    high_complexity_funcs = []

    for file_path in py_files:
        if "__pycache__" in str(file_path):
            continue
            
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            lines = content.split('\n')
            total_metrics["files"] += 1
            total_metrics["total_lines"] += len(lines)
            total_metrics["code_lines"] += len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            total_metrics["functions"] += len(functions)
            total_metrics["classes"] += len(classes)
            total_metrics["imports"] += len(imports)
            
            for func in functions:
                if ast.get_docstring(func):
                    total_metrics["documented"] += 1
                complexity = _calculate_complexity(func)
                if complexity > max_complexity:
                    high_complexity_funcs.append((str(file_path), func.name, complexity))
                    
        except Exception:
            continue

    doc_coverage = (total_metrics["documented"] / total_metrics["functions"] * 100) if total_metrics["functions"] else 100

    report = [
        f"Directory Analysis: {p}",
        "=" * 50,
        f"Files analyzed: {total_metrics['files']}",
        f"Total lines: {total_metrics['total_lines']}",
        f"Code lines: {total_metrics['code_lines']}",
        f"Functions: {total_metrics['functions']}",
        f"Classes: {total_metrics['classes']}",
        f"Documentation coverage: {doc_coverage:.1f}%",
    ]

    if high_complexity_funcs:
        report.extend([
            "",
            f"High complexity functions (>{max_complexity}):",
        ])
        for file_path, func_name, complexity in sorted(high_complexity_funcs, key=lambda x: x[2], reverse=True)[:10]:
            report.append(f"  - {file_path}:{func_name}: {complexity}")

    return "\n".join(report)


def _get_metrics(p: Path) -> str:
    """Get quick metrics for a file or directory."""
    if p.is_file():
        return _analyze_file(p, 10)
    else:
        return _analyze_directory(p, 10)


def _find_issues(p: Path) -> str:
    """Find common code issues."""
    issues = []
    
    if p.is_file():
        files = [p] if str(p).endswith('.py') else []
    else:
        files = [f for f in p.rglob("*.py") if "__pycache__" not in str(f)]

    for file_path in files:
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            # Check for common issues
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Bare except
                if re.match(r'^\s*except\s*:', stripped):
                    issues.append((str(file_path), i, "Bare except clause", line.strip()[:50]))
                
                # Print statements (potential debug code)
                if re.match(r'^\s*print\s*\(', stripped) and '"__main__"' not in content:
                    issues.append((str(file_path), i, "Print statement", line.strip()[:50]))
                
                # TODO/FIXME comments
                if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE):
                    issues.append((str(file_path), i, "TODO/FIXME comment", line.strip()[:50]))
                
                # Long lines
                if len(line) > 100:
                    issues.append((str(file_path), i, "Long line (>100 chars)", line[:50] + "..."))
                    
        except Exception:
            continue

    if not issues:
        return f"No common issues found in {p}"

    report = [f"Issues found in {p}:", "=" * 50]
    for file_path, line, issue_type, snippet in issues[:30]:
        report.append(f"{file_path}:{line}: [{issue_type}] {snippet}")
    
    if len(issues) > 30:
        report.append(f"\n... and {len(issues) - 30} more issues")

    return "\n".join(report)


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity for a function."""
    complexity = 1
    
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            complexity += 1
    
    return complexity
