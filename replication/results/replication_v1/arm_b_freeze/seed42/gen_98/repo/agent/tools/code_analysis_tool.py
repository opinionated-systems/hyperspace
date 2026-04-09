"""
Code analysis tool: analyze Python code structure and extract useful information.

Provides functionality to:
- Extract function/class definitions
- Analyze imports and dependencies
- Get code metrics (line counts, complexity)
- Find function/class references
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code structure. "
            "Commands: list_functions, list_classes, get_imports, analyze_file, find_references. "
            "Helps understand code organization and dependencies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["list_functions", "list_classes", "get_imports", "analyze_file", "find_references"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file or directory.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (function/class name for find_references).",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
) -> str:
    """Execute a code analysis command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {p} does not exist."

        if command == "list_functions":
            return _list_functions(p)
        elif command == "list_classes":
            return _list_classes(p)
        elif command == "get_imports":
            return _get_imports(p)
        elif command == "analyze_file":
            return _analyze_file(p)
        elif command == "find_references":
            if pattern is None:
                return "Error: pattern required for find_references."
            return _find_references(p, pattern)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _parse_file(p: Path) -> ast.AST | None:
    """Parse a Python file and return the AST."""
    try:
        content = p.read_text()
        return ast.parse(content)
    except SyntaxError as e:
        return None
    except Exception:
        return None


def _list_functions(p: Path) -> str:
    """List all function definitions in a file or directory."""
    functions = []
    
    if p.is_file() and p.suffix == ".py":
        tree = _parse_file(p)
        if tree is None:
            return f"Error: Could not parse {p}"
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function signature
                args = []
                for arg in node.args.args:
                    args.append(arg.arg)
                for arg in node.args.kwonlyargs:
                    args.append(arg.arg)
                if node.args.vararg:
                    args.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    args.append(f"**{node.args.kwarg.arg}")
                
                sig = f"{node.name}({', '.join(args)})"
                doc = ast.get_docstring(node)
                doc_preview = f" - {doc[:60]}..." if doc else ""
                functions.append(f"  Line {node.lineno}: {sig}{doc_preview}")
    
    elif p.is_dir():
        for root, _, files in os.walk(p):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = Path(root) / filename
                    tree = _parse_file(filepath)
                    if tree:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                functions.append(f"  {filepath}:{node.lineno}: {node.name}()")
    
    if functions:
        return f"Functions found ({len(functions)}):\n" + "\n".join(functions[:50])
    return "No functions found."


def _list_classes(p: Path) -> str:
    """List all class definitions in a file or directory."""
    classes = []
    
    if p.is_file() and p.suffix == ".py":
        tree = _parse_file(p)
        if tree is None:
            return f"Error: Could not parse {p}"
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Get base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(base.attr)
                
                base_str = f"({', '.join(bases)})" if bases else ""
                doc = ast.get_docstring(node)
                doc_preview = f" - {doc[:60]}..." if doc else ""
                classes.append(f"  Line {node.lineno}: {node.name}{base_str}{doc_preview}")
    
    elif p.is_dir():
        for root, _, files in os.walk(p):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = Path(root) / filename
                    tree = _parse_file(filepath)
                    if tree:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                classes.append(f"  {filepath}:{node.lineno}: {node.name}")
    
    if classes:
        return f"Classes found ({len(classes)}):\n" + "\n".join(classes[:50])
    return "No classes found."


def _get_imports(p: Path) -> str:
    """Get all imports from a file or directory."""
    imports = []
    
    if p.is_file() and p.suffix == ".py":
        tree = _parse_file(p)
        if tree is None:
            return f"Error: Could not parse {p}"
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"  Line {node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(f"  Line {node.lineno}: from {module} import {', '.join(names)}")
    
    elif p.is_dir():
        all_imports = {}
        for root, _, files in os.walk(p):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = Path(root) / filename
                    tree = _parse_file(filepath)
                    if tree:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    all_imports.setdefault(alias.name, []).append(str(filepath))
                            elif isinstance(node, ast.ImportFrom):
                                module = node.module or ""
                                all_imports.setdefault(module, []).append(str(filepath))
        
        for name, files in sorted(all_imports.items()):
            imports.append(f"  {name}: used in {len(files)} file(s)")
    
    if imports:
        return f"Imports found ({len(imports)}):\n" + "\n".join(imports[:50])
    return "No imports found."


def _analyze_file(p: Path) -> str:
    """Analyze a single Python file and provide metrics."""
    if not p.is_file() or p.suffix != ".py":
        return "Error: analyze_file requires a Python file path."
    
    try:
        content = p.read_text()
        lines = content.split("\n")
        tree = ast.parse(content)
        
        # Count various metrics
        total_lines = len(lines)
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        comment_lines = len([l for l in lines if l.strip().startswith("#")])
        blank_lines = len([l for l in lines if not l.strip()])
        
        func_count = 0
        class_count = 0
        import_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
        
        result = [
            f"Analysis of {p}:",
            f"  Total lines: {total_lines}",
            f"  Code lines: {code_lines}",
            f"  Comment lines: {comment_lines}",
            f"  Blank lines: {blank_lines}",
            f"  Functions: {func_count}",
            f"  Classes: {class_count}",
            f"  Import statements: {import_count}",
        ]
        
        return "\n".join(result)
    except SyntaxError as e:
        return f"Syntax error in {p}: {e}"
    except Exception as e:
        return f"Error analyzing {p}: {e}"


def _find_references(p: Path, pattern: str) -> str:
    """Find references to a function or class name in a file or directory."""
    references = []
    
    if p.is_file() and p.suffix == ".py":
        content = p.read_text()
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if pattern in line:
                references.append(f"  Line {i}: {line.strip()[:100]}")
    
    elif p.is_dir():
        for root, _, files in os.walk(p):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = Path(root) / filename
                    try:
                        content = filepath.read_text()
                        lines = content.split("\n")
                        for i, line in enumerate(lines, 1):
                            if pattern in line:
                                references.append(f"  {filepath}:{i}: {line.strip()[:80]}")
                    except Exception:
                        continue
    
    if references:
        return f"References to '{pattern}' ({len(references)}):\n" + "\n".join(references[:50])
    return f"No references to '{pattern}' found."
