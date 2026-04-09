"""
Code analysis tool: parse Python files to understand structure.

Provides insights into classes, functions, imports, and dependencies
to help the agent make more informed modifications.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Any

# Simple LRU cache for file analysis results
_analysis_cache: dict[str, tuple[str, float, str]] = {}  # path -> (content_hash, mtime, result)
_MAX_CACHE_SIZE = 100


def _get_cache_key(p: Path) -> str:
    """Generate cache key from file path, content hash, and mtime."""
    try:
        stat = p.stat()
        content = p.read_bytes()
        content_hash = hashlib.md5(content).hexdigest()
        return f"{p}:{content_hash}:{stat.st_mtime}"
    except Exception:
        return None


def _get_cached_result(p: Path) -> str | None:
    """Get cached analysis result if file hasn't changed."""
    cache_key = _get_cache_key(p)
    if cache_key is None:
        return None
    return _analysis_cache.get(cache_key)


def _set_cached_result(p: Path, result: str) -> None:
    """Cache analysis result."""
    global _analysis_cache
    cache_key = _get_cache_key(p)
    if cache_key is None:
        return
    
    # Simple LRU: clear if too big
    if len(_analysis_cache) >= _MAX_CACHE_SIZE:
        _analysis_cache.clear()
    
    _analysis_cache[cache_key] = result


def tool_info() -> dict:
    return {
        "name": "code_analysis",
        "description": (
            "Analyze Python code structure to understand classes, functions, "
            "imports, and dependencies. Helps make informed code modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["analyze_file", "analyze_module", "find_dependencies", "get_outline"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file or directory.",
                },
                "target": {
                    "type": "string",
                    "description": "Target name for specific lookups (optional).",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(
    command: str,
    path: str,
    target: str | None = None,
) -> str:
    """Execute a code analysis command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if command == "analyze_file":
            return _analyze_file(p)
        elif command == "analyze_module":
            return _analyze_module(p)
        elif command == "find_dependencies":
            return _find_dependencies(p)
        elif command == "get_outline":
            return _get_outline(p, target)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _analyze_file(p: Path) -> str:
    """Analyze a single Python file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    if not p.is_file():
        return f"Error: {p} is not a file."
    if not str(p).endswith('.py'):
        return f"Error: {p} is not a Python file."

    # Check cache first
    cached = _get_cached_result(p)
    if cached:
        return cached

    try:
        content = p.read_text()
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error parsing {p}: {e}"
    except Exception as e:
        return f"Error reading {p}: {e}"

    # Extract information
    imports = []
    classes = []
    functions = []
    top_level_calls = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(f"{module}: {', '.join(names)}")
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(_get_func_signature(item))
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "bases": [ast.unparse(base) for base in node.bases] if hasattr(ast, 'unparse') else [],
            })
        elif isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "signature": _get_func_signature(node),
            })
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            top_level_calls.append(ast.unparse(node.value) if hasattr(ast, 'unparse') else "call")

    # Format output
    lines = [f"Analysis of {p}:", "=" * 50]
    
    if imports:
        lines.extend(["", f"Imports ({len(imports)}):"])
        for imp in imports[:20]:
            lines.append(f"  - {imp}")
        if len(imports) > 20:
            lines.append(f"  ... and {len(imports) - 20} more")

    if classes:
        lines.extend(["", f"Classes ({len(classes)}):"])
        for cls in classes:
            bases = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
            lines.append(f"  - {cls['name']}{bases} [line {cls['line']}]")
            for method in cls['methods'][:5]:
                lines.append(f"      {method}")
            if len(cls['methods']) > 5:
                lines.append(f"      ... and {len(cls['methods']) - 5} more methods")

    if functions:
        lines.extend(["", f"Functions ({len(functions)}):"])
        for func in functions[:15]:
            lines.append(f"  - {func['signature']} [line {func['line']}]")
        if len(functions) > 15:
            lines.append(f"  ... and {len(functions) - 15} more")

    if top_level_calls:
        lines.extend(["", f"Top-level calls ({len(top_level_calls)}):"])
        for call in top_level_calls[:5]:
            lines.append(f"  - {call[:80]}")

    stats = _get_file_stats(content)
    lines.extend(["", "Statistics:", f"  Lines: {stats['lines']}", f"  Code lines: {stats['code_lines']}",
                  f"  Comment lines: {stats['comment_lines']}", f"  Blank lines: {stats['blank_lines']}"])

    result = "\n".join(lines)
    _set_cached_result(p, result)
    return result


def _get_func_signature(node: ast.FunctionDef) -> str:
    """Extract function signature."""
    args = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation and hasattr(ast, 'unparse'):
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)
    
    # Add defaults info
    defaults_start = len(node.args.args) - len(node.args.defaults)
    for i, default in enumerate(node.args.defaults):
        if hasattr(ast, 'unparse'):
            args[defaults_start + i] += f"={ast.unparse(default)}"
    
    # Add *args and **kwargs
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    
    returns = ""
    if node.returns and hasattr(ast, 'unparse'):
        returns = f" -> {ast.unparse(node.returns)}"
    
    return f"def {node.name}({', '.join(args)}){returns}"


def _get_file_stats(content: str) -> dict:
    """Get file statistics."""
    lines = content.split('\n')
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith('#'):
            comment_lines += 1
        else:
            code_lines += 1
    
    return {
        "lines": len(lines),
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
    }


def _analyze_module(p: Path) -> str:
    """Analyze a Python module (directory)."""
    if not p.exists():
        return f"Error: {p} does not exist."
    if not p.is_dir():
        return f"Error: {p} is not a directory."

    py_files = list(p.rglob("*.py"))
    if not py_files:
        return f"No Python files found in {p}"

    # Count items
    total_classes = 0
    total_functions = 0
    total_imports = 0
    file_count = len(py_files)

    for f in py_files:
        try:
            content = f.read_text()
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    total_imports += 1
                elif isinstance(node, ast.ClassDef):
                    total_classes += 1
                elif isinstance(node, ast.FunctionDef):
                    total_functions += 1
        except:
            pass

    lines = [
        f"Module analysis for {p}:",
        "=" * 50,
        f"",
        f"Python files: {file_count}",
        f"Classes: {total_classes}",
        f"Functions: {total_functions}",
        f"Import statements: {total_imports}",
        f"",
        "Files:",
    ]

    for f in sorted(py_files)[:30]:
        rel_path = f.relative_to(p)
        lines.append(f"  - {rel_path}")
    if len(py_files) > 30:
        lines.append(f"  ... and {len(py_files) - 30} more files")

    return "\n".join(lines)


def _find_dependencies(p: Path) -> str:
    """Find all dependencies (imports) in a file or module."""
    if not p.exists():
        return f"Error: {p} does not exist."

    imports = set()
    
    if p.is_file():
        files = [p] if str(p).endswith('.py') else []
    else:
        files = list(p.rglob("*.py"))

    for f in files:
        try:
            content = f.read_text()
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except:
            pass

    if not imports:
        return f"No external dependencies found in {p}"

    # Categorize
    stdlib = {'os', 'sys', 'pathlib', 'json', 're', 'time', 'datetime', 'collections', 'typing', 
              'functools', 'itertools', 'math', 'random', 'hashlib', 'base64', 'io', 'csv',
              'ast', 'inspect', 'importlib', 'types', 'warnings', 'logging', 'threading',
              'concurrent', 'asyncio', 'unittest', 'contextlib', 'dataclasses', 'enum'}
    
    stdlib_imports = sorted(imports & stdlib)
    external_imports = sorted(imports - stdlib)

    lines = [f"Dependencies for {p}:", "=" * 50, ""]
    
    if stdlib_imports:
        lines.extend([f"Standard library ({len(stdlib_imports)}):", ""])
        for imp in stdlib_imports:
            lines.append(f"  - {imp}")
        lines.append("")
    
    if external_imports:
        lines.extend([f"External packages ({len(external_imports)}):", ""])
        for imp in external_imports:
            lines.append(f"  - {imp}")

    return "\n".join(lines)


def _get_outline(p: Path, target: str | None = None) -> str:
    """Get a structural outline of a Python file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    if not p.is_file():
        return f"Error: {p} is not a file."
    if not str(p).endswith('.py'):
        return f"Error: {p} is not a Python file."

    try:
        content = p.read_text()
        tree = ast.parse(content)
    except Exception as e:
        return f"Error parsing {p}: {e}"

    lines = [f"Outline of {p}:", "=" * 50, ""]

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if target is None or node.name == target:
                lines.append(f"Class {node.name} [line {node.lineno}]")
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        decorator = "@staticmethod " if any(
                            isinstance(d, ast.Name) and d.id == 'staticmethod' 
                            for d in item.decorator_list
                        ) else ""
                        lines.append(f"    {decorator}{item.name}() [line {item.lineno}]")
        elif isinstance(node, ast.FunctionDef):
            if target is None or node.name == target:
                lines.append(f"Function {node.name}() [line {node.lineno}]")

    if len(lines) == 3:
        if target:
            return f"No definition found for '{target}' in {p}"
        return f"No classes or functions found in {p}"

    return "\n".join(lines)
