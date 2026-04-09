"""
Code analysis tool: analyze Python files for structure, complexity, and imports.

Provides insights into code organization, function/class counts,
cyclomatic complexity estimation, and dependency analysis.
Useful for understanding codebase structure before making modifications.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set allowed root directory for file operations."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_allowed_path(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code structure, complexity, and dependencies. "
            "Returns function/class counts, import analysis, and complexity metrics. "
            "Useful for understanding code before modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the Python file or directory to analyze.",
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Include complexity metrics (default: true).",
                },
            },
            "required": ["path"],
        },
    }


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to collect code metrics."""
    
    def __init__(self) -> None:
        self.functions: list[dict] = []
        self.classes: list[dict] = []
        self.imports: list[str] = []
        self.from_imports: list[dict] = []
        self.docstrings: int = 0
        self.comments: int = 0
        self.lines_of_code: int = 0
        self.blank_lines: int = 0
        self._current_class: str | None = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        func_info = {
            "name": node.name,
            "line": node.lineno,
            "args": len(node.args.args) + len(node.args.kwonlyargs),
            "decorators": len(node.decorator_list),
            "class": self._current_class,
            "complexity": self._estimate_complexity(node),
            "has_docstring": ast.get_docstring(node) is not None,
        }
        self.functions.append(func_info)
        if func_info["has_docstring"]:
            self.docstrings += 1
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self.visit_FunctionDef(node)  # type: ignore
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        prev_class = self._current_class
        self._current_class = node.name
        
        class_info = {
            "name": node.name,
            "line": node.lineno,
            "bases": [self._get_name(base) for base in node.bases],
            "methods": 0,
            "has_docstring": ast.get_docstring(node) is not None,
        }
        
        # Count methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                class_info["methods"] += 1
                
        self.classes.append(class_info)
        if class_info["has_docstring"]:
            self.docstrings += 1
            
        self.generic_visit(node)
        self._current_class = prev_class
        
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from import statement."""
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.from_imports.append({"module": module, "names": names})
        self.generic_visit(node)
        
    def _estimate_complexity(self, node: ast.AST) -> int:
        """Estimate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        return complexity
        
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "<unknown>"


def _analyze_file(filepath: str) -> dict[str, Any]:
    """Analyze a single Python file."""
    try:
        content = Path(filepath).read_text()
        lines = content.split("\n")
        
        # Count lines
        loc = len([l for l in lines if l.strip()])
        blank = len([l for l in lines if not l.strip()])
        comments = len([l for l in lines if l.strip().startswith("#")])
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
            
        analyzer = CodeAnalyzer()
        analyzer.lines_of_code = loc
        analyzer.blank_lines = blank
        analyzer.comments = comments
        analyzer.visit(tree)
        
        # Calculate metrics
        total_complexity = sum(f["complexity"] for f in analyzer.functions)
        avg_complexity = total_complexity / len(analyzer.functions) if analyzer.functions else 0
        
        # Find high complexity functions
        high_complexity = [f for f in analyzer.functions if f["complexity"] > 10]
        
        return {
            "file": filepath,
            "lines": {
                "total": len(lines),
                "code": loc,
                "blank": blank,
                "comments": comments,
            },
            "functions": {
                "count": len(analyzer.functions),
                "list": analyzer.functions,
                "total_complexity": total_complexity,
                "average_complexity": round(avg_complexity, 2),
                "high_complexity_functions": high_complexity,
            },
            "classes": {
                "count": len(analyzer.classes),
                "list": analyzer.classes,
            },
            "imports": {
                "direct": analyzer.imports,
                "from": analyzer.from_imports,
            },
            "docstrings": analyzer.docstrings,
        }
        
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}


def _format_analysis(result: dict) -> str:
    """Format analysis result as readable string."""
    if "error" in result:
        return f"Error analyzing {result.get('file', 'unknown')}: {result['error']}"
        
    lines = [
        f"Code Analysis: {result['file']}",
        "=" * 50,
        "",
        "📊 Line Counts:",
        f"  Total: {result['lines']['total']}",
        f"  Code: {result['lines']['code']}",
        f"  Blank: {result['lines']['blank']}",
        f"  Comments: {result['lines']['comments']}",
        "",
        f"📦 Classes: {result['classes']['count']}",
    ]
    
    for cls in result['classes']['list']:
        bases = f" ({', '.join(cls['bases'])})" if cls['bases'] else ""
        doc = " 📄" if cls['has_docstring'] else ""
        lines.append(f"  - {cls['name']}{bases} - {cls['methods']} methods{doc}")
        
    lines.extend([
        "",
        f"🔧 Functions: {result['functions']['count']}",
        f"  Total complexity: {result['functions']['total_complexity']}",
        f"  Average complexity: {result['functions']['average_complexity']}",
    ])
    
    for func in result['functions']['list'][:10]:  # Show first 10
        prefix = "  - "
        if func['class']:
            prefix = f"    {func['name']}"
        else:
            prefix = f"  - {func['name']}"
        doc = " 📄" if func['has_docstring'] else ""
        lines.append(f"{prefix} (complexity: {func['complexity']}, args: {func['args']}){doc}")
        
    if len(result['functions']['list']) > 10:
        lines.append(f"  ... and {len(result['functions']['list']) - 10} more functions")
        
    if result['functions']['high_complexity_functions']:
        lines.extend([
            "",
            "⚠️ High Complexity Functions (>10):",
        ])
        for func in result['functions']['high_complexity_functions']:
            lines.append(f"  - {func['name']}: complexity {func['complexity']}")
            
    if result['imports']['direct'] or result['imports']['from']:
        lines.extend([
            "",
            "📥 Imports:",
        ])
        for imp in result['imports']['direct'][:10]:
            lines.append(f"  - import {imp}")
        for imp in result['imports']['from'][:10]:
            names = ', '.join(imp['names'][:5])
            if len(imp['names']) > 5:
                names += f" (+{len(imp['names']) - 5} more)"
            lines.append(f"  - from {imp['module']} import {names}")
            
    lines.extend([
        "",
        f"📄 Docstrings: {result['docstrings']}/{result['functions']['count'] + result['classes']['count']} items",
    ])
    
    return "\n".join(lines)


def tool_function(path: str, include_metrics: bool = True) -> str:
    """Analyze Python code structure and return metrics.
    
    Args:
        path: Path to Python file or directory
        include_metrics: Whether to include complexity metrics
        
    Returns:
        Formatted analysis report
    """
    if not _is_allowed_path(path):
        return f"Error: Path '{path}' is outside allowed root."
        
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Path '{path}' does not exist."
            
        if p.is_file():
            if not path.endswith('.py'):
                return f"Error: File '{path}' is not a Python file."
            result = _analyze_file(str(p))
            return _format_analysis(result)
            
        elif p.is_dir():
            # Analyze all Python files in directory
            results = []
            for py_file in p.rglob('*.py'):
                if '__pycache__' not in str(py_file):
                    result = _analyze_file(str(py_file))
                    if 'error' not in result:
                        results.append(result)
                        
            if not results:
                return f"No Python files found in '{path}'"
                
            # Aggregate stats
            total_files = len(results)
            total_lines = sum(r['lines']['total'] for r in results)
            total_functions = sum(r['functions']['count'] for r in results)
            total_classes = sum(r['classes']['count'] for r in results)
            total_complexity = sum(r['functions']['total_complexity'] for r in results)
            
            lines = [
                f"Directory Analysis: {path}",
                "=" * 50,
                "",
                f"📁 Python files: {total_files}",
                f"📄 Total lines: {total_lines}",
                f"📦 Total classes: {total_classes}",
                f"🔧 Total functions: {total_functions}",
                f"📊 Total complexity: {total_complexity}",
                "",
                "Files analyzed:",
            ]
            
            for r in sorted(results, key=lambda x: x['lines']['total'], reverse=True)[:20]:
                rel_path = os.path.relpath(r['file'], path)
                lines.append(f"  - {rel_path}: {r['lines']['total']} lines, "
                           f"{r['functions']['count']} funcs, "
                           f"{r['classes']['count']} classes")
                           
            if len(results) > 20:
                lines.append(f"  ... and {len(results) - 20} more files")
                
            return "\n".join(lines)
            
    except Exception as e:
        return f"Error analyzing '{path}': {e}"
