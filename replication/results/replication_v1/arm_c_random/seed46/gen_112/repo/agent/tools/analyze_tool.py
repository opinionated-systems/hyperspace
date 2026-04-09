"""
Code analysis tool: parse and analyze Python code structure.

Provides AST-based analysis to help the meta agent understand
code structure, find functions, classes, and their signatures.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "analyze",
        "description": (
            "Analyze Python code structure using AST parsing. "
            "Commands: list_functions, list_classes, get_function_source, get_signature. "
            "Helps understand code structure without reading entire files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["list_functions", "list_classes", "get_function_source", "get_signature"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the Python file.",
                },
                "name": {
                    "type": "string",
                    "description": "Function or class name (for get_function_source, get_signature).",
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


def _truncate(content: str, max_len: int = 5000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def _check_path(path: str) -> tuple[Path | None, str]:
    """Validate and return the path, or return error message."""
    p = Path(path)
    
    if not p.is_absolute():
        return None, f"Error: {path} is not an absolute path."
    
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(_ALLOWED_ROOT):
            return None, f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    
    if not p.exists():
        return None, f"Error: {p} does not exist."
    
    if not p.is_file():
        return None, f"Error: {p} is not a file."
    
    return p, ""


def _parse_file(p: Path) -> tuple[ast.AST | None, str]:
    """Parse a Python file and return the AST."""
    try:
        content = p.read_text()
        tree = ast.parse(content)
        return tree, content
    except SyntaxError as e:
        return None, f"Syntax error in {p}: {e}"
    except UnicodeDecodeError:
        return None, f"Error: {p} is not a text file"
    except Exception as e:
        return None, f"Error reading {p}: {e}"


def _get_node_source(content: str, node: ast.AST) -> str:
    """Extract source code for an AST node."""
    lines = content.split("\n")
    start_line = node.lineno - 1
    end_line = node.end_lineno
    return "\n".join(lines[start_line:end_line])


def _format_function_info(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Extract function information from AST node."""
    args = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)
    
    # Handle *args and **kwargs
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    
    returns = ""
    if node.returns:
        returns = f" -> {ast.unparse(node.returns)}"
    
    decorators = [ast.unparse(d) for d in node.decorator_list]
    
    return {
        "name": node.name,
        "args": args,
        "returns": returns,
        "decorators": decorators,
        "line": node.lineno,
        "docstring": ast.get_docstring(node),
    }


def _format_class_info(node: ast.ClassDef) -> dict[str, Any]:
    """Extract class information from AST node."""
    bases = [ast.unparse(b) for b in node.bases]
    methods = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_format_function_info(item))
    
    return {
        "name": node.name,
        "bases": bases,
        "line": node.lineno,
        "docstring": ast.get_docstring(node),
        "methods": methods,
    }


def tool_function(
    command: str,
    path: str,
    name: str | None = None,
) -> str:
    """Execute a code analysis command."""
    p, error = _check_path(path)
    if error:
        return error
    
    tree, content = _parse_file(p)
    if tree is None:
        return content  # Error message
    
    if command == "list_functions":
        return _list_functions(tree, content, str(p))
    elif command == "list_classes":
        return _list_classes(tree, content, str(p))
    elif command == "get_function_source":
        if not name:
            return "Error: 'name' parameter required for get_function_source"
        return _get_function_source(tree, content, name)
    elif command == "get_signature":
        if not name:
            return "Error: 'name' parameter required for get_signature"
        return _get_signature(tree, content, name)
    else:
        return f"Error: unknown command {command}"


def _list_functions(tree: ast.AST, content: str, path: str) -> str:
    """List all functions in the file."""
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info = _format_function_info(node)
            sig = f"def {info['name']}({', '.join(info['args'])}){info['returns']}"
            functions.append(f"  Line {info['line']}: {sig}")
    
    if not functions:
        return f"No functions found in {path}"
    
    return f"Functions in {path}:\n" + "\n".join(sorted(functions, key=lambda x: int(x.split()[1].rstrip(':'))))


def _list_classes(tree: ast.AST, content: str, path: str) -> str:
    """List all classes in the file."""
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            info = _format_class_info(node)
            bases = f"({', '.join(info['bases'])})" if info['bases'] else ""
            classes.append(f"  Line {info['line']}: class {info['name']}{bases}")
    
    if not classes:
        return f"No classes found in {path}"
    
    return f"Classes in {path}:\n" + "\n".join(sorted(classes, key=lambda x: int(x.split()[1].rstrip(':'))))


def _get_function_source(tree: ast.AST, content: str, name: str) -> str:
    """Get the source code of a specific function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            source = _get_node_source(content, node)
            return f"Source of {name}:\n```python\n{source}\n```"
    
    # Also check methods in classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    source = _get_node_source(content, item)
                    return f"Source of {node.name}.{name}:\n```python\n{source}\n```"
    
    return f"Function '{name}' not found"


def _get_signature(tree: ast.AST, content: str, name: str) -> str:
    """Get the signature of a specific function or class."""
    # Check for function
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            info = _format_function_info(node)
            sig = f"def {info['name']}({', '.join(info['args'])}){info['returns']}"
            result = f"Signature: {sig}\nLine: {info['line']}"
            if info['docstring']:
                result += f"\nDocstring: {info['docstring'][:200]}"
            return result
    
    # Check for class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            info = _format_class_info(node)
            bases = f"({', '.join(info['bases'])})" if info['bases'] else ""
            result = f"Signature: class {info['name']}{bases}\nLine: {info['line']}"
            if info['docstring']:
                result += f"\nDocstring: {info['docstring'][:200]}"
            if info['methods']:
                result += f"\nMethods: {len(info['methods'])}"
            return result
    
    # Check for method in class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    info = _format_function_info(item)
                    sig = f"def {info['name']}({', '.join(info['args'])}){info['returns']}"
                    return f"Signature: {node.name}.{sig}\nLine: {info['line']} (in class {node.name})"
    
    return f"'{name}' not found (function, method, or class)"
