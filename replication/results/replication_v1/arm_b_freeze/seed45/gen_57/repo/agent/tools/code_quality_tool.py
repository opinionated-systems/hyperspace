"""
Code quality analysis tool for identifying potential issues in Python code.

Provides static analysis capabilities to help the meta agent identify
areas for improvement in the codebase.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _analyze_complexity(source: str, filename: str) -> list[dict]:
    """Analyze code complexity metrics."""
    issues = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [{"type": "syntax_error", "message": str(e), "line": e.lineno}]
    
    # Count lines and functions
    total_lines = len(source.splitlines())
    function_count = 0
    class_count = 0
    long_functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_count += 1
            # Check function length
            if hasattr(node, 'end_lineno') and node.end_lineno:
                length = node.end_lineno - node.lineno
                if length > 50:
                    long_functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "length": length
                    })
        elif isinstance(node, ast.ClassDef):
            class_count += 1
    
    if long_functions:
        issues.append({
            "type": "long_functions",
            "message": f"Found {len(long_functions)} functions longer than 50 lines",
            "details": long_functions[:5]  # Limit to first 5
        })
    
    return issues


def _analyze_patterns(source: str, filename: str) -> list[dict]:
    """Analyze code for common anti-patterns."""
    issues = []
    lines = source.splitlines()
    
    # Check for bare except clauses
    for i, line in enumerate(lines, 1):
        if re.search(r'except\s*:', line) and 'except Exception' not in line:
            issues.append({
                "type": "bare_except",
                "message": f"Bare except clause found at line {i}",
                "line": i,
                "severity": "warning"
            })
    
    # Check for TODO/FIXME comments
    todo_count = 0
    for i, line in enumerate(lines, 1):
        if re.search(r'#.*\b(TODO|FIXME|XXX|HACK)\b', line, re.IGNORECASE):
            todo_count += 1
    
    if todo_count > 0:
        issues.append({
            "type": "todo_comments",
            "message": f"Found {todo_count} TODO/FIXME/XXX comments",
            "count": todo_count,
            "severity": "info"
        })
    
    # Check for print statements (should use logging)
    print_count = 0
    for i, line in enumerate(lines, 1):
        if re.search(r'^\s*print\s*\(', line) and not line.strip().startswith('#'):
            print_count += 1
    
    if print_count > 0:
        issues.append({
            "type": "print_statements",
            "message": f"Found {print_count} print statements (consider using logging)",
            "count": print_count,
            "severity": "info"
        })
    
    # Check for mutable default arguments
    mutable_default_pattern = re.compile(r'def\s+\w+\s*\([^)]*=\s*(\[|\{)')
    for i, line in enumerate(lines, 1):
        if mutable_default_pattern.search(line):
            issues.append({
                "type": "mutable_default",
                "message": f"Potential mutable default argument at line {i}",
                "line": i,
                "severity": "warning"
            })
    
    return issues


def _analyze_imports(source: str, filename: str) -> list[dict]:
    """Analyze import patterns."""
    issues = []
    
    # Check for wildcard imports
    if re.search(r'from\s+\S+\s+import\s+\*', source):
        issues.append({
            "type": "wildcard_import",
            "message": "Wildcard import found (import *)",
            "severity": "warning"
        })
    
    # Check for unused imports (simple check)
    try:
        tree = ast.parse(source)
        imported_names = set()
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Simple heuristic - may have false positives
        potentially_unused = imported_names - used_names - {'__future__'}
        if potentially_unused:
            issues.append({
                "type": "potentially_unused_imports",
                "message": f"Potentially unused imports: {', '.join(sorted(potentially_unused))}",
                "imports": list(potentially_unused),
                "severity": "info"
            })
    except SyntaxError:
        pass
    
    return issues


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "code_quality",
        "description": "Analyze Python code for quality issues, anti-patterns, and potential improvements. Returns a report of findings including complexity metrics, code smells, and style issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the Python file or directory to analyze"
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to analyze (default: 10)",
                    "default": 10
                }
            },
            "required": ["path"]
        }
    }


def tool_function(path: str, max_files: int = 10) -> str:
    """Analyze code quality in the given path.
    
    Args:
        path: Path to Python file or directory
        max_files: Maximum files to analyze
        
    Returns:
        Formatted report of code quality issues
    """
    target = Path(path)
    
    if not target.exists():
        return f"Error: Path not found: {path}"
    
    files_to_analyze = []
    
    if target.is_file():
        if target.suffix == '.py':
            files_to_analyze.append(target)
    else:
        # Find Python files, excluding __pycache__ and hidden files
        for py_file in target.rglob('*.py'):
            if '__pycache__' not in str(py_file) and not py_file.name.startswith('.'):
                files_to_analyze.append(py_file)
                if len(files_to_analyze) >= max_files:
                    break
    
    if not files_to_analyze:
        return f"No Python files found in {path}"
    
    all_results = []
    
    for file_path in files_to_analyze:
        try:
            source = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = str(file_path.relative_to(target)) if target.is_dir() else str(file_path.name)
            
            file_issues = {
                "file": rel_path,
                "lines": len(source.splitlines()),
                "issues": []
            }
            
            # Run all analyses
            file_issues["issues"].extend(_analyze_complexity(source, rel_path))
            file_issues["issues"].extend(_analyze_patterns(source, rel_path))
            file_issues["issues"].extend(_analyze_imports(source, rel_path))
            
            if file_issues["issues"]:
                all_results.append(file_issues)
                
        except Exception as e:
            all_results.append({
                "file": str(file_path),
                "error": str(e)
            })
    
    # Format the report
    if not all_results:
        return f"No quality issues found in {len(files_to_analyze)} file(s) analyzed."
    
    report_lines = [
        f"Code Quality Analysis Report",
        f"============================",
        f"",
        f"Analyzed {len(files_to_analyze)} file(s), found issues in {len(all_results)} file(s):",
        f""
    ]
    
    for result in all_results:
        report_lines.append(f"File: {result['file']} ({result.get('lines', '?')} lines)")
        report_lines.append("-" * 40)
        
        if "error" in result:
            report_lines.append(f"  ERROR: {result['error']}")
        else:
            for issue in result.get("issues", []):
                severity = issue.get("severity", "info").upper()
                msg = issue.get("message", "Unknown issue")
                report_lines.append(f"  [{severity}] {msg}")
                
                # Add details for certain issue types
                if "details" in issue and issue["details"]:
                    for detail in issue["details"][:3]:  # Limit details
                        if isinstance(detail, dict):
                            detail_str = ", ".join(f"{k}={v}" for k, v in detail.items())
                            report_lines.append(f"    - {detail_str}")
        
        report_lines.append("")
    
    report_lines.append("Summary:")
    total_issues = sum(len(r.get("issues", [])) for r in all_results if "issues" in r)
    report_lines.append(f"  - Total issues found: {total_issues}")
    report_lines.append(f"  - Files with issues: {len(all_results)}")
    report_lines.append(f"")
    report_lines.append("Recommendations:")
    report_lines.append("  - Address 'warning' severity issues first")
    report_lines.append("  - Consider refactoring long functions (>50 lines)")
    report_lines.append("  - Replace print statements with proper logging")
    report_lines.append("  - Fix bare except clauses to catch specific exceptions")
    
    return "\n".join(report_lines)
