"""
Code analysis tool: analyze Python code structure and extract useful information.

Provides capabilities for:
- Extracting function/class definitions
- Finding imports and dependencies
- Analyzing code complexity metrics
- Identifying potential issues
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def _analyze_python_file(file_path: str) -> dict[str, Any]:
    """Analyze a Python file and extract structural information.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if not path.suffix == '.py':
            return {"error": f"Not a Python file: {file_path}"}
        
        content = path.read_text(encoding="utf-8", errors="ignore")
        
        # Parse the AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"Syntax error in file: {e}"}
        
        # Extract information
        functions = []
        classes = []
        imports = []
        docstrings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                }
                functions.append(func_info)
                if func_info["docstring"]:
                    docstrings.append({
                        "type": "function",
                        "name": node.name,
                        "docstring": func_info["docstring"][:200],
                    })
            
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    "docstring": ast.get_docstring(node),
                }
                classes.append(class_info)
                if class_info["docstring"]:
                    docstrings.append({
                        "type": "class",
                        "name": node.name,
                        "docstring": class_info["docstring"][:200],
                    })
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "name": alias.name,
                        "alias": alias.asname,
                    })
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                    })
        
        # Calculate basic metrics
        lines = content.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        
        return {
            "file_path": str(path),
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "code_lines": len(code_lines),
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "docstrings": docstrings,
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports),
        }
        
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}


def _extract_code_snippet(file_path: str, line_start: int, line_end: int | None = None) -> dict[str, Any]:
    """Extract a code snippet from a file.
    
    Args:
        file_path: Path to the file
        line_start: Starting line number (1-indexed)
        line_end: Ending line number (optional, defaults to line_start + 20)
        
    Returns:
        Dictionary with the code snippet
    """
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split('\n')
        
        if line_start < 1:
            line_start = 1
        if line_end is None:
            line_end = min(line_start + 20, len(lines))
        if line_end > len(lines):
            line_end = len(lines)
        
        snippet_lines = lines[line_start - 1:line_end]
        
        # Add line numbers
        numbered_lines = []
        for i, line in enumerate(snippet_lines, start=line_start):
            numbered_lines.append(f"{i:4d}: {line}")
        
        return {
            "file_path": str(path),
            "line_start": line_start,
            "line_end": line_end,
            "snippet": '\n'.join(numbered_lines),
            "total_lines": len(lines),
        }
        
    except Exception as e:
        return {"error": f"Extraction failed: {e}"}


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "code_analysis",
        "description": "Analyze Python code structure to extract functions, classes, imports, and metrics. Also supports extracting specific code snippets by line number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute: 'analyze' for full file analysis, 'snippet' for code extraction",
                    "enum": ["analyze", "snippet"],
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the Python file to analyze",
                },
                "line_start": {
                    "type": "integer",
                    "description": "Starting line number for snippet extraction (1-indexed, only for 'snippet' command)",
                },
                "line_end": {
                    "type": "integer",
                    "description": "Ending line number for snippet extraction (optional, only for 'snippet' command)",
                },
            },
            "required": ["command", "file_path"],
        },
    }


def tool_function(
    command: str,
    file_path: str,
    line_start: int | None = None,
    line_end: int | None = None,
) -> str:
    """Execute the code analysis tool."""
    if command == "analyze":
        result = _analyze_python_file(file_path)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        lines = [
            f"Analysis of {result['file_path']}:",
            "",
            f"Lines: {result['total_lines']} total, {result['non_empty_lines']} non-empty, {result['code_lines']} code",
            f"Functions: {result['function_count']}, Classes: {result['class_count']}, Imports: {result['import_count']}",
            "",
        ]
        
        if result['imports']:
            lines.append("Imports:")
            for imp in result['imports'][:20]:  # Limit to 20 imports
                if imp['type'] == 'import':
                    alias = f" as {imp['alias']}" if imp['alias'] else ""
                    lines.append(f"  - import {imp['name']}{alias}")
                else:
                    alias = f" as {imp['alias']}" if imp['alias'] else ""
                    lines.append(f"  - from {imp['module']} import {imp['name']}{alias}")
            if len(result['imports']) > 20:
                lines.append(f"  ... and {len(result['imports']) - 20} more imports")
            lines.append("")
        
        if result['classes']:
            lines.append("Classes:")
            for cls in result['classes']:
                method_count = len(cls['methods'])
                doc_info = " (has docstring)" if cls['docstring'] else ""
                lines.append(f"  - {cls['name']} (line {cls['line']}, {method_count} methods){doc_info}")
            lines.append("")
        
        if result['functions']:
            lines.append("Functions:")
            for func in result['functions'][:30]:  # Limit to 30 functions
                arg_str = ', '.join(func['args'])
                doc_info = " (has docstring)" if func['docstring'] else ""
                lines.append(f"  - {func['name']}({arg_str}) (line {func['line']}){doc_info}")
            if len(result['functions']) > 30:
                lines.append(f"  ... and {len(result['functions']) - 30} more functions")
            lines.append("")
        
        return '\n'.join(lines)
    
    elif command == "snippet":
        if line_start is None:
            return "Error: line_start is required for snippet command"
        
        result = _extract_code_snippet(file_path, line_start, line_end)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Code snippet from {result['file_path']} (lines {result['line_start']}-{result['line_end']} of {result['total_lines']}):\n\n{result['snippet']}"
    
    else:
        return f"Error: Unknown command '{command}'. Use 'analyze' or 'snippet'."
