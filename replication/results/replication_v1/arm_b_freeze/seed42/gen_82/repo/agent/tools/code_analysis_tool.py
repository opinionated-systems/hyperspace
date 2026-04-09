"""
Code analysis tool: analyze Python code structure and statistics.

Provides capabilities for understanding codebase structure including:
- Counting lines of code, functions, classes
- Finding definitions and dependencies
- Analyzing file complexity metrics
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def _count_lines(filepath: str) -> dict[str, int]:
    """Count total, code, comment, and blank lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return {"error": str(e)}
    
    total = len(lines)
    code = 0
    comments = 0
    blank = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
        elif stripped.startswith('#'):
            comments += 1
        else:
            code += 1
    
    return {
        "total": total,
        "code": code,
        "comments": comments,
        "blank": blank,
    }


def _analyze_python_file(filepath: str) -> dict[str, Any]:
    """Parse a Python file and extract structural information."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        return {"error": str(e)}
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}
    
    functions = []
    classes = []
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "args": len(node.args.args),
            })
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body 
                if isinstance(n, ast.FunctionDef)
            ]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(f"{module}.{names[0]}" if module else names[0])
    
    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "function_count": len(functions),
        "class_count": len(classes),
        "import_count": len(imports),
    }


def _analyze_directory(dirpath: str, max_depth: int = 3) -> dict[str, Any]:
    """Analyze a directory structure and summarize Python files."""
    path = Path(dirpath)
    if not path.exists():
        return {"error": f"Directory not found: {dirpath}"}
    
    if not path.is_dir():
        return {"error": f"Not a directory: {dirpath}"}
    
    python_files = []
    total_lines = 0
    total_functions = 0
    total_classes = 0
    
    for root, dirs, files in os.walk(path):
        # Limit depth
        depth = root.count(os.sep) - str(path).count(os.sep)
        if depth >= max_depth:
            del dirs[:]
            continue
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, path)
                
                line_counts = _count_lines(filepath)
                if "error" not in line_counts:
                    total_lines += line_counts.get("total", 0)
                
                structure = _analyze_python_file(filepath)
                if "error" not in structure:
                    total_functions += structure.get("function_count", 0)
                    total_classes += structure.get("class_count", 0)
                
                python_files.append({
                    "path": rel_path,
                    "lines": line_counts.get("total", 0),
                    "functions": structure.get("function_count", 0),
                    "classes": structure.get("class_count", 0),
                })
    
    return {
        "file_count": len(python_files),
        "total_lines": total_lines,
        "total_functions": total_functions,
        "total_classes": total_classes,
        "files": sorted(python_files, key=lambda x: x["lines"], reverse=True)[:20],
    }


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "code_analysis",
        "description": "Analyze Python code structure and statistics. Can count lines, find functions/classes, analyze imports, and summarize directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["file_stats", "structure", "directory_summary"],
                    "description": "Type of analysis to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Path to file or directory to analyze",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth for directory_summary (default: 3)",
                    "default": 3,
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str, max_depth: int = 3) -> str:
    """Execute code analysis based on command."""
    if command == "file_stats":
        result = _count_lines(path)
        if "error" in result:
            return f"Error: {result['error']}"
        return (
            f"File: {path}\n"
            f"  Total lines: {result['total']}\n"
            f"  Code lines: {result['code']}\n"
            f"  Comments: {result['comments']}\n"
            f"  Blank lines: {result['blank']}"
        )
    
    elif command == "structure":
        result = _analyze_python_file(path)
        if "error" in result:
            return f"Error: {result['error']}"
        
        output = [f"Structure of {path}:", f"  Functions ({result['function_count']}):"]
        for func in result['functions']:
            output.append(f"    - {func['name']} (line {func['line']}, {func['args']} args)")
        
        output.append(f"  Classes ({result['class_count']}):")
        for cls in result['classes']:
            methods = ', '.join(cls['methods'][:5])
            if len(cls['methods']) > 5:
                methods += f" ... ({len(cls['methods'])-5} more)"
            output.append(f"    - {cls['name']} (line {cls['line']}, methods: {methods})")
        
        output.append(f"  Imports ({result['import_count']}):")
        for imp in result['imports'][:10]:
            output.append(f"    - {imp}")
        if len(result['imports']) > 10:
            output.append(f"    ... ({len(result['imports'])-10} more)")
        
        return '\n'.join(output)
    
    elif command == "directory_summary":
        result = _analyze_directory(path, max_depth)
        if "error" in result:
            return f"Error: {result['error']}"
        
        output = [
            f"Directory summary for {path}:",
            f"  Python files: {result['file_count']}",
            f"  Total lines: {result['total_lines']}",
            f"  Total functions: {result['total_functions']}",
            f"  Total classes: {result['total_classes']}",
            "",
            "Top files by size:",
        ]
        for file in result['files']:
            output.append(
                f"  - {file['path']}: {file['lines']} lines, "
                f"{file['functions']} funcs, {file['classes']} classes"
            )
        
        return '\n'.join(output)
    
    else:
        return f"Unknown command: {command}. Use 'file_stats', 'structure', or 'directory_summary'."
