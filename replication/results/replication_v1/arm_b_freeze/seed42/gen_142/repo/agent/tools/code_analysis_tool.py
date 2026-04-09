"""
Code analysis tool: extract function/class definitions and analyze code structure.

Provides capabilities to understand code before making modifications:
- Extract function signatures and docstrings
- List class methods and attributes
- Analyze imports and dependencies
- Calculate basic code metrics
"""

from __future__ import annotations

import ast
import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code structure to extract functions, classes, "
            "and their signatures. Useful for understanding code before modifications. "
            "Commands: list_functions, list_classes, get_function_source, analyze_file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["list_functions", "list_classes", "get_function_source", "analyze_file"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Python file to analyze.",
                },
                "name": {
                    "type": "string",
                    "description": "Function or class name (for get_function_source).",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(
    command: str,
    path: str,
    name: str | None = None,
) -> str:
    """Execute a code analysis command."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if not p.is_file():
            return f"Error: {p} is not a file."
        
        if not str(p).endswith('.py'):
            return f"Error: {p} is not a Python file."
        
        # Read and parse the file
        try:
            content = p.read_text()
            tree = ast.parse(content)
        except SyntaxError as e:
            return f"Error: Syntax error in {p}: {e}"
        except Exception as e:
            return f"Error: Could not parse {p}: {e}"
        
        if command == "list_functions":
            return _list_functions(tree, p, content)
        elif command == "list_classes":
            return _list_classes(tree, p, content)
        elif command == "get_function_source":
            if name is None:
                return "Error: 'name' parameter required for get_function_source."
            return _get_function_source(tree, p, content, name)
        elif command == "analyze_file":
            return _analyze_file(tree, p, content)
        else:
            return f"Error: unknown command {command}"
            
    except Exception as e:
        return f"Error: {e}"


def _get_source_lines(content: str, node: ast.AST) -> str:
    """Extract source code for an AST node."""
    lines = content.split('\n')
    start_line = node.lineno - 1
    end_line = getattr(node, 'end_lineno', node.lineno)
    if end_line:
        end_line = end_line - 1
    return '\n'.join(lines[start_line:end_line + 1])


def _get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract decorator names from a node."""
    decorators = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            decorators.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            decorators.append(f"{decorator.attr}")
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                decorators.append(f"{decorator.id}()")
            elif isinstance(decorator.func, ast.Attribute):
                decorators.append(f"{decorator.func.attr}()")
    return decorators


def _format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, content: str) -> str:
    """Format a function signature with decorators."""
    decorators = _get_decorators(node)
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    
    # Get the function signature line
    sig_line = _get_source_lines(content, node).split('\n')[0]
    
    result = []
    for dec in decorators:
        result.append(f"@{dec}")
    result.append(sig_line)
    
    return '\n'.join(result)


def _list_functions(tree: ast.AST, path: Path, content: str) -> str:
    """List all functions in the file."""
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods (functions inside classes)
            if isinstance(getattr(node, 'parent', None), ast.ClassDef):
                continue
            
            name = node.name
            is_async = isinstance(node, ast.AsyncFunctionDef)
            args_count = len(node.args.args) + len(node.args.kwonlyargs)
            if node.args.vararg:
                args_count += 1
            if node.args.kwarg:
                args_count += 1
            
            # Get docstring
            docstring = ast.get_docstring(node)
            doc_summary = docstring.split('\n')[0][:60] if docstring else "No docstring"
            
            functions.append({
                "name": name,
                "async": is_async,
                "args": args_count,
                "line": node.lineno,
                "doc": doc_summary,
            })
    
    if not functions:
        return f"No top-level functions found in {path}"
    
    lines = [f"Functions in {path}:", ""]
    for func in functions:
        async_prefix = "async " if func["async"] else ""
        lines.append(f"  {async_prefix}def {func['name']}()  (line {func['line']}, {func['args']} args)")
        lines.append(f"    Doc: {func['doc']}")
    
    return '\n'.join(lines)


def _list_classes(tree: ast.AST, path: Path, content: str) -> str:
    """List all classes in the file."""
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Count methods
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
            
            # Get base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{base.attr}")
            
            # Get docstring
            docstring = ast.get_docstring(node)
            doc_summary = docstring.split('\n')[0][:60] if docstring else "No docstring"
            
            classes.append({
                "name": node.name,
                "bases": bases,
                "methods": methods,
                "line": node.lineno,
                "doc": doc_summary,
            })
    
    if not classes:
        return f"No classes found in {path}"
    
    lines = [f"Classes in {path}:", ""]
    for cls in classes:
        bases_str = f"({', '.join(cls['bases'])})" if cls["bases"] else ""
        lines.append(f"  class {cls['name']}{bases_str}  (line {cls['line']})")
        lines.append(f"    Methods: {', '.join(cls['methods']) if cls['methods'] else 'None'}")
        lines.append(f"    Doc: {cls['doc']}")
    
    return '\n'.join(lines)


def _get_function_source(tree: ast.AST, path: Path, content: str, name: str) -> str:
    """Get the source code of a specific function or class."""
    # First look for functions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            source = _get_source_lines(content, node)
            return f"Source of {name} in {path} (line {node.lineno}):\n\n{source}"
    
    # Then look for classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            source = _get_source_lines(content, node)
            return f"Source of class {name} in {path} (line {node.lineno}):\n\n{source}"
    
    return f"Error: Function or class '{name}' not found in {path}"


def _analyze_file(tree: ast.AST, path: Path, content: str) -> str:
    """Provide a comprehensive analysis of the file."""
    lines = content.split('\n')
    total_lines = len(lines)
    
    # Count different types of nodes
    functions = []
    classes = []
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not isinstance(getattr(node, 'parent', None), ast.ClassDef):
                functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(f"{module}: {', '.join(names)}")
    
    # Calculate complexity metrics
    non_empty_lines = len([l for l in lines if l.strip()])
    comment_lines = len([l for l in lines if l.strip().startswith('#')])
    
    result = [
        f"Analysis of {path}:",
        "",
        f"  Total lines: {total_lines}",
        f"  Non-empty lines: {non_empty_lines}",
        f"  Comment lines: {comment_lines}",
        f"  Functions: {len(functions)}",
        f"  Classes: {len(classes)}",
        f"  Imports: {len(imports)}",
        "",
    ]
    
    if functions:
        result.append(f"  Function names: {', '.join(functions)}")
    if classes:
        result.append(f"  Class names: {', '.join(classes)}")
    if imports:
        result.append(f"  Imports: {', '.join(imports[:10])}")
        if len(imports) > 10:
            result.append(f"    ... and {len(imports) - 10} more")
    
    return '\n'.join(result)
