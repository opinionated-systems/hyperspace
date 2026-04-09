"""
Code analysis tool: analyze Python code structure and extract useful information.

Provides capabilities to understand code structure, find function/class definitions,
and analyze imports and dependencies.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "analyze",
        "description": (
            "Analyze Python code structure to extract functions, classes, imports, "
            "and other code elements. Useful for understanding codebase structure "
            "before making modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["functions", "classes", "imports", "structure", "complexity"],
                    "description": "Analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file to analyze.",
                },
            },
            "required": ["command", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict analysis operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    command: str,
    path: str,
) -> str:
    """Execute a code analysis command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Analysis restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {p} does not exist."

        if p.is_dir():
            return f"Error: {p} is a directory. Please specify a Python file."

        if not str(p).endswith('.py'):
            return f"Error: {p} is not a Python file."

        try:
            content = p.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except SyntaxError as e:
            return f"Error: Syntax error in {p}: {e}"
        except Exception as e:
            return f"Error: Could not parse {p}: {e}"

        if command == "functions":
            return _analyze_functions(tree, str(p))
        elif command == "classes":
            return _analyze_classes(tree, str(p))
        elif command == "imports":
            return _analyze_imports(tree, str(p))
        elif command == "structure":
            return _analyze_structure(tree, str(p))
        elif command == "complexity":
            return _analyze_complexity(tree, str(p))
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _analyze_functions(tree: ast.AST, path: str) -> str:
    """Extract all function definitions from the AST."""
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function signature
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)
            
            # Handle defaults
            defaults_start = len(args) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                args[defaults_start + i] += f" = {ast.unparse(default)}"
            
            # Handle *args and **kwargs
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")
            
            # Get return annotation
            returns = ""
            if node.returns:
                returns = f" -> {ast.unparse(node.returns)}"
            
            # Get docstring
            docstring = ast.get_docstring(node)
            doc_summary = ""
            if docstring:
                doc_summary = docstring.split('\n')[0][:80]
                if len(docstring) > 80:
                    doc_summary += "..."
            
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "signature": f"({', '.join(args)}){returns}",
                "docstring": doc_summary,
            })
    
    if not functions:
        return f"No functions found in {path}"
    
    output = f"Functions in {path}:\n\n"
    for func in functions:
        output += f"  Line {func['line']}: def {func['name']}{func['signature']}\n"
        if func['docstring']:
            output += f"    Doc: {func['docstring']}\n"
    
    return output


def _analyze_classes(tree: ast.AST, path: str) -> str:
    """Extract all class definitions from the AST."""
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get base classes
            bases = [ast.unparse(base) for base in node.bases]
            
            # Get docstring
            docstring = ast.get_docstring(node)
            doc_summary = ""
            if docstring:
                doc_summary = docstring.split('\n')[0][:80]
                if len(docstring) > 80:
                    doc_summary += "..."
            
            # Count methods
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "bases": bases,
                "methods": len(methods),
                "docstring": doc_summary,
            })
    
    if not classes:
        return f"No classes found in {path}"
    
    output = f"Classes in {path}:\n\n"
    for cls in classes:
        base_str = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
        output += f"  Line {cls['line']}: class {cls['name']}{base_str}\n"
        output += f"    Methods: {cls['methods']}\n"
        if cls['docstring']:
            output += f"    Doc: {cls['docstring']}\n"
    
    return output


def _analyze_imports(tree: ast.AST, path: str) -> str:
    """Extract all imports from the AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "type": "from",
                    "module": module,
                    "name": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                })
    
    if not imports:
        return f"No imports found in {path}"
    
    output = f"Imports in {path}:\n\n"
    for imp in imports:
        if imp["type"] == "import":
            alias_str = f" as {imp['alias']}" if imp["alias"] else ""
            output += f"  Line {imp['line']}: import {imp['module']}{alias_str}\n"
        else:
            alias_str = f" as {imp['alias']}" if imp["alias"] else ""
            output += f"  Line {imp['line']}: from {imp['module']} import {imp['name']}{alias_str}\n"
    
    return output


def _analyze_structure(tree: ast.AST, path: str) -> str:
    """Provide a high-level overview of the file structure."""
    stats = {
        "functions": 0,
        "classes": 0,
        "imports": 0,
        "docstrings": 0,
        "lines": 0,
    }
    
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            stats["functions"] += 1
            if ast.get_docstring(node):
                stats["docstrings"] += 1
            top_level.append(f"Function: {node.name} (line {node.lineno})")
        elif isinstance(node, ast.ClassDef):
            stats["classes"] += 1
            if ast.get_docstring(node):
                stats["docstrings"] += 1
            methods = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
            top_level.append(f"Class: {node.name} (line {node.lineno}, {methods} methods)")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            stats["imports"] += 1
    
    # Count total lines
    try:
        content = Path(path).read_text(encoding='utf-8')
        stats["lines"] = len(content.splitlines())
    except:
        pass
    
    output = f"Structure of {path}:\n\n"
    output += f"Statistics:\n"
    output += f"  Total lines: {stats['lines']}\n"
    output += f"  Imports: {stats['imports']}\n"
    output += f"  Classes: {stats['classes']}\n"
    output += f"  Functions: {stats['functions']}\n"
    output += f"  Documented: {stats['docstrings']}\n\n"
    
    if top_level:
        output += "Top-level definitions:\n"
        for item in top_level:
            output += f"  - {item}\n"
    
    return output


def _analyze_complexity(tree: ast.AST, path: str) -> str:
    """Analyze code complexity metrics."""
    complexities = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Simple cyclomatic complexity approximation
            complexity = 1  # Base complexity
            
            # Count branches
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
                elif isinstance(child, ast.comprehension):
                    complexity += 1
            
            # Count lines in function
            if node.body:
                end_line = node.end_lineno or node.lineno
                lines = end_line - node.lineno
            else:
                lines = 0
            
            complexities.append({
                "name": node.name,
                "line": node.lineno,
                "complexity": complexity,
                "lines": lines,
            })
    
    if not complexities:
        return f"No functions to analyze in {path}"
    
    # Sort by complexity
    complexities.sort(key=lambda x: x["complexity"], reverse=True)
    
    output = f"Complexity analysis for {path}:\n\n"
    output += "Functions by cyclomatic complexity:\n"
    for func in complexities:
        risk = "low"
        if func["complexity"] > 10:
            risk = "high"
        elif func["complexity"] > 5:
            risk = "medium"
        
        output += f"  Line {func['line']}: {func['name']} - "
        output += f"complexity: {func['complexity']} ({risk} risk), "
        output += f"lines: {func['lines']}\n"
    
    avg_complexity = sum(c["complexity"] for c in complexities) / len(complexities)
    output += f"\nAverage complexity: {avg_complexity:.1f}\n"
    
    return output
