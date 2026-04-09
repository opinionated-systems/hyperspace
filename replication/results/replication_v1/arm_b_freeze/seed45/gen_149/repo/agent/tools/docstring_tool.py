"""
Docstring analyzer tool: analyze and improve Python docstrings.

Provides capabilities to:
- Check for missing docstrings on functions and classes
- Analyze docstring quality (completeness, style)
- Suggest improvements based on Google/NumPy/PEP 257 conventions
- Check parameter documentation completeness

This helps the meta agent improve code documentation quality.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "docstring_analyzer",
        "description": (
            "Analyze Python docstrings for completeness and quality. "
            "Detects missing docstrings, incomplete parameter documentation, "
            "and style issues. Suggests improvements following Google/NumPy/PEP 257 conventions. "
            "Helps improve code documentation quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file or directory to analyze.",
                },
                "style": {
                    "type": "string",
                    "enum": ["google", "numpy", "pep257"],
                    "description": "Docstring style convention to check against (default: google).",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to analyze in a directory (default: 20).",
                },
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict docstring analysis operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Analysis restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_list(items: list[str], max_items: int = 20) -> list[str]:
    """Truncate list to max_items."""
    if len(items) > max_items:
        return items[:max_items] + [f"... ({len(items) - max_items} more items)"]
    return items


def tool_function(
    path: str,
    style: str = "google",
    max_files: int = 20,
) -> str:
    """Analyze docstrings at the given path."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if p.is_dir():
            return _analyze_directory(p, style, max_files)
        elif p.suffix == ".py":
            return _analyze_file(p, style)
        else:
            return f"Error: {p} is not a Python file or directory."
            
    except Exception as e:
        return f"Error: {e}"


def _analyze_directory(path: Path, style: str, max_files: int) -> str:
    """Analyze all Python files in a directory."""
    py_files = list(path.rglob("*.py"))
    
    # Exclude common non-source directories
    exclude_patterns = [
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".pytest_cache", ".mypy_cache", ".tox", "build", "dist"
    ]
    py_files = [
        f for f in py_files 
        if not any(pattern in str(f) for pattern in exclude_patterns)
    ]
    
    if not py_files:
        return f"No Python files found in {path}"
    
    # Sort and limit
    py_files.sort()
    total_files = len(py_files)
    py_files = py_files[:max_files]
    
    results = []
    total_issues = 0
    files_with_issues = 0
    
    for file_path in py_files:
        file_result = _analyze_file_internal(file_path, style)
        if file_result["issues"]:
            total_issues += len(file_result["issues"])
            files_with_issues += 1
            results.append(f"\n{'='*60}")
            results.append(f"File: {file_path}")
            results.append(f"{'='*60}")
            for issue in file_result["issues"]:
                results.append(f"  • {issue}")
    
    if not results:
        return f"✓ Analyzed {min(total_files, max_files)} Python files - all docstrings look good!"
    
    summary = f"Docstring Analysis Summary for {path}:\n"
    summary += f"  Files analyzed: {len(py_files)} of {total_files}\n"
    summary += f"  Files with issues: {files_with_issues}\n"
    summary += f"  Total issues: {total_issues}\n"
    summary += "\n".join(results)
    
    return summary


def _analyze_file(path: Path, style: str) -> str:
    """Analyze a single Python file."""
    result = _analyze_file_internal(path, style)
    
    output = [f"Docstring Analysis for {path}:", "=" * 60]
    
    if result["issues"]:
        output.append("\nIssues Found:")
        for issue in result["issues"]:
            output.append(f"  • {issue}")
    else:
        output.append("\n✓ No docstring issues found!")
    
    output.append(f"\nSummary:")
    output.append(f"  Functions/classes checked: {result['stats']['total']}")
    output.append(f"  With docstrings: {result['stats']['with_docstring']}")
    output.append(f"  Missing docstrings: {result['stats']['missing']}")
    output.append(f"  Incomplete docstrings: {result['stats']['incomplete']}")
    
    return "\n".join(output)


def _analyze_file_internal(path: Path, style: str) -> dict[str, Any]:
    """Internal analysis function returning structured data."""
    issues = []
    stats = {
        "total": 0,
        "with_docstring": 0,
        "missing": 0,
        "incomplete": 0,
    }
    
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"issues": [f"Error reading file: {e}"], "stats": stats}
    
    # Check for syntax errors
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        issues.append(f"Syntax Error at line {e.lineno}: {e.msg}")
        return {"issues": issues, "stats": stats}
    except Exception as e:
        issues.append(f"Parse Error: {e}")
        return {"issues": issues, "stats": stats}
    
    # Analyze all function and class definitions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_issues, node_stats = _analyze_function(node, source, style)
            issues.extend(node_issues)
            _update_stats(stats, node_stats)
        elif isinstance(node, ast.ClassDef):
            node_issues, node_stats = _analyze_class(node, source, style)
            issues.extend(node_issues)
            _update_stats(stats, node_stats)
    
    return {"issues": _truncate_list(issues, 30), "stats": stats}


def _update_stats(stats: dict, node_stats: dict) -> None:
    """Update aggregate stats with node stats."""
    for key in stats:
        stats[key] += node_stats.get(key, 0)


def _analyze_function(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str, style: str) -> tuple[list[str], dict]:
    """Analyze a function's docstring."""
    issues = []
    stats = {"total": 1, "with_docstring": 0, "missing": 0, "incomplete": 0}
    
    # Skip private functions (starting with _)
    if node.name.startswith("_") and not node.name.startswith("__"):
        return [], {"total": 0, "with_docstring": 0, "missing": 0, "incomplete": 0}
    
    # Skip dunder methods that don't need docstrings
    if node.name in ("__init__", "__repr__", "__str__", "__eq__", "__hash__"):
        return [], {"total": 0, "with_docstring": 0, "missing": 0, "incomplete": 0}
    
    docstring = ast.get_docstring(node)
    
    if not docstring:
        # Check if it's a property with getter
        if any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list):
            issues.append(f"Property '{node.name}' at line {node.lineno} is missing a docstring")
        else:
            issues.append(f"Function '{node.name}' at line {node.lineno} is missing a docstring")
        stats["missing"] = 1
        return issues, stats
    
    stats["with_docstring"] = 1
    
    # Check docstring quality
    quality_issues = _check_docstring_quality(docstring, node, style)
    if quality_issues:
        issues.extend([f"Function '{node.name}' at line {node.lineno}: {issue}" for issue in quality_issues])
        stats["incomplete"] = 1
    
    return issues, stats


def _analyze_class(node: ast.ClassDef, source: str, style: str) -> tuple[list[str], dict]:
    """Analyze a class's docstring."""
    issues = []
    stats = {"total": 1, "with_docstring": 0, "missing": 0, "incomplete": 0}
    
    # Skip private classes
    if node.name.startswith("_"):
        return [], {"total": 0, "with_docstring": 0, "missing": 0, "incomplete": 0}
    
    docstring = ast.get_docstring(node)
    
    if not docstring:
        issues.append(f"Class '{node.name}' at line {node.lineno} is missing a docstring")
        stats["missing"] = 1
        return issues, stats
    
    stats["with_docstring"] = 1
    
    # Check docstring quality
    quality_issues = _check_docstring_quality(docstring, node, style)
    if quality_issues:
        issues.extend([f"Class '{node.name}' at line {node.lineno}: {issue}" for issue in quality_issues])
        stats["incomplete"] = 1
    
    return issues, stats


def _check_docstring_quality(docstring: str, node: ast.AST, style: str) -> list[str]:
    """Check docstring quality and completeness."""
    issues = []
    
    # Check for very short docstrings
    if len(docstring.strip()) < 10:
        issues.append("Docstring is too short (less than 10 characters)")
    
    # Check for one-line docstrings that should be multi-line
    if "\n" not in docstring and len(docstring) > 80:
        issues.append("Long one-line docstring should be formatted as multi-line")
    
    # Check for proper capitalization
    stripped = docstring.strip()
    if stripped and not stripped[0].isupper():
        issues.append("Docstring should start with a capital letter")
    
    # Check for proper ending punctuation
    if stripped and not stripped[-1] in ".!?":
        issues.append("Docstring should end with proper punctuation")
    
    # Check function-specific requirements
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Check for Args section if function has parameters
        has_args = bool(node.args.args or node.args.kwonlyargs or node.args.posonlyargs)
        if has_args and not _node_is_property(node):
            if style == "google":
                if "Args:" not in docstring and "Arguments:" not in docstring and "Parameters:" not in docstring:
                    issues.append("Missing 'Args:' section for function with parameters")
            elif style == "numpy":
                if "Parameters" not in docstring:
                    issues.append("Missing 'Parameters' section for function with parameters")
        
        # Check for Returns section if function has a return statement
        if _has_return_statement(node) and not _node_is_property(node):
            if style == "google":
                if "Returns:" not in docstring:
                    issues.append("Missing 'Returns:' section for function with return statement")
            elif style == "numpy":
                if "Returns" not in docstring:
                    issues.append("Missing 'Returns' section for function with return statement")
        
        # Check for Raises section if function has raise statement
        if _has_raise_statement(node):
            if style == "google":
                if "Raises:" not in docstring:
                    issues.append("Consider adding 'Raises:' section for function that raises exceptions")
            elif style == "numpy":
                if "Raises" not in docstring:
                    issues.append("Consider adding 'Raises' section for function that raises exceptions")
    
    # Check class-specific requirements
    if isinstance(node, ast.ClassDef):
        # Check for Attributes section if class has instance attributes
        if _has_instance_attributes(node) and style == "google":
            if "Attributes:" not in docstring:
                issues.append("Consider adding 'Attributes:' section for class with instance attributes")
    
    return issues


def _node_is_property(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function node is decorated as a property."""
    return any(
        isinstance(d, ast.Name) and d.id == "property"
        for d in node.decorator_list
    )


def _has_return_statement(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function has a return statement."""
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            return True
    return False


def _has_raise_statement(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function has a raise statement."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Raise, ast.Assert)):
            return True
    return False


def _has_instance_attributes(node: ast.ClassDef) -> bool:
    """Check if a class defines instance attributes in __init__."""
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            # Look for self.attribute assignments
            for subchild in ast.walk(child):
                if isinstance(subchild, ast.Attribute):
                    if isinstance(subchild.value, ast.Name) and subchild.value.id == "self":
                        return True
    return False
