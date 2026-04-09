"""
Code analysis tool: analyze Python file structure, metrics, and dependencies.

Provides capabilities to understand code structure before making modifications.
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
            "Analyze Python code structure, metrics, and dependencies. "
            "Commands: analyze_file, analyze_directory, find_dependencies, get_metrics. "
            "Helps understand code before making modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["analyze_file", "analyze_directory", "find_dependencies", "get_metrics"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional pattern to filter results.",
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

        if command == "analyze_file":
            return _analyze_file(p)
        elif command == "analyze_directory":
            return _analyze_directory(p, pattern)
        elif command == "find_dependencies":
            return _find_dependencies(p)
        elif command == "get_metrics":
            return _get_metrics(p)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _analyze_file(p: Path) -> str:
    """Analyze a single Python file for classes, functions, and imports."""
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
        return f"Error parsing {p}: {e}"
    except Exception as e:
        return f"Error reading {p}: {e}"

    imports = []
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(f"{module}: {', '.join(names)}")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "docstring": ast.get_docstring(node),
            })
        elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
            # Only top-level functions (not methods)
            if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                })

    lines = content.split('\n')
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    
    result = [f"Analysis of {p}:", "=" * 50]
    result.append(f"Total lines: {total_lines}")
    result.append(f"Code lines: {code_lines}")
    result.append("")
    
    if imports:
        result.append(f"Imports ({len(imports)}):")
        for imp in imports[:15]:  # Limit to first 15
            result.append(f"  - {imp}")
        if len(imports) > 15:
            result.append(f"  ... and {len(imports) - 15} more")
        result.append("")
    
    if classes:
        result.append(f"Classes ({len(classes)}):")
        for cls in classes:
            result.append(f"  - {cls['name']} (line {cls['line']})")
            if cls['methods']:
                result.append(f"    Methods: {', '.join(cls['methods'][:5])}")
                if len(cls['methods']) > 5:
                    result.append(f"    ... and {len(cls['methods']) - 5} more methods")
        result.append("")
    
    if functions:
        result.append(f"Top-level functions ({len(functions)}):")
        for func in functions[:10]:
            result.append(f"  - {func['name']}({', '.join(func['args'])}) (line {func['line']})")
        if len(functions) > 10:
            result.append(f"  ... and {len(functions) - 10} more")
    
    return '\n'.join(result)


def _analyze_directory(p: Path, pattern: str | None = None) -> str:
    """Analyze all Python files in a directory."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_dir():
        return f"Error: {p} is not a directory."

    py_files = list(p.rglob("*.py"))
    py_files = [f for f in py_files if not any(part.startswith('.') or part == '__pycache__' for part in f.parts)]
    
    if pattern:
        py_files = [f for f in py_files if pattern in str(f)]
    
    if not py_files:
        return f"No Python files found in {p}"

    total_lines = 0
    total_files = len(py_files)
    file_summaries = []
    
    for f in py_files:
        try:
            content = f.read_text()
            lines = content.split('\n')
            total_lines += len(lines)
            
            # Quick parse for summary
            try:
                tree = ast.parse(content)
                classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                file_summaries.append(f"  {f.relative_to(p)}: {len(lines)} lines, {classes} classes, {functions} functions")
            except:
                file_summaries.append(f"  {f.relative_to(p)}: {len(lines)} lines (parse error)")
        except Exception as e:
            file_summaries.append(f"  {f.relative_to(p)}: error - {e}")

    result = [
        f"Directory analysis of {p}:",
        "=" * 50,
        f"Total Python files: {total_files}",
        f"Total lines of code: {total_lines}",
        "",
        "Files:",
    ]
    result.extend(file_summaries[:30])  # Limit output
    if len(file_summaries) > 30:
        result.append(f"  ... and {len(file_summaries) - 30} more files")
    
    return '\n'.join(result)


def _find_dependencies(p: Path) -> str:
    """Find import dependencies for a Python file or directory."""
    if not p.exists():
        return f"Error: {p} does not exist."

    if p.is_file():
        files = [p] if str(p).endswith('.py') else []
    else:
        files = [f for f in p.rglob("*.py") if not any(part.startswith('.') or part == '__pycache__' for part in f.parts)]

    if not files:
        return f"No Python files found at {p}"

    all_imports = set()
    local_imports = set()
    external_imports = set()
    
    for f in files:
        try:
            content = f.read_text()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        all_imports.add(alias.name)
                        if alias.name.startswith('.') or alias.name.startswith(str(p.stem)):
                            local_imports.add(alias.name)
                        else:
                            external_imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        all_imports.add(node.module)
                        if node.module.startswith('.') or str(p) in node.module:
                            local_imports.add(node.module)
                        else:
                            external_imports.add(node.module.split('.')[0])
        except:
            continue

    result = [
        f"Dependency analysis for {p}:",
        "=" * 50,
        f"Total unique imports: {len(all_imports)}",
        "",
        f"External dependencies ({len(external_imports)}):",
    ]
    result.extend(f"  - {imp}" for imp in sorted(external_imports)[:20])
    if len(external_imports) > 20:
        result.append(f"  ... and {len(external_imports) - 20} more")
    
    if local_imports:
        result.extend(["", f"Local imports ({len(local_imports)}):"])
        result.extend(f"  - {imp}" for imp in sorted(local_imports))
    
    return '\n'.join(result)


def _get_metrics(p: Path) -> str:
    """Get code metrics for a file or directory."""
    if not p.exists():
        return f"Error: {p} does not exist."

    if p.is_file():
        files = [p] if str(p).endswith('.py') else []
    else:
        files = [f for f in p.rglob("*.py") if not any(part.startswith('.') or part == '__pycache__' for part in f.parts)]

    if not files:
        return f"No Python files found at {p}"

    total_lines = 0
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    total_classes = 0
    total_functions = 0
    
    for f in files:
        try:
            content = f.read_text()
            lines = content.split('\n')
            total_lines += len(lines)
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    blank_lines += 1
                elif stripped.startswith('#'):
                    comment_lines += 1
                else:
                    code_lines += 1
            
            try:
                tree = ast.parse(content)
                total_classes += len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                total_functions += len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            except:
                pass
        except:
            continue

    result = [
        f"Code metrics for {p}:",
        "=" * 50,
        f"Files analyzed: {len(files)}",
        "",
        "Line counts:",
        f"  Total lines: {total_lines}",
        f"  Code lines: {code_lines}",
        f"  Comment lines: {comment_lines}",
        f"  Blank lines: {blank_lines}",
        "",
        "Structure:",
        f"  Classes: {total_classes}",
        f"  Functions: {total_functions}",
        "",
        "Averages per file:",
        f"  Lines: {total_lines // len(files) if files else 0}",
        f"  Classes: {total_classes / len(files):.1f}" if files else "  Classes: 0",
        f"  Functions: {total_functions / len(files):.1f}" if files else "  Functions: 0",
    ]
    
    return '\n'.join(result)
