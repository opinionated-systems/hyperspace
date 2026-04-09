"""
File summary tool: quickly summarize file contents and statistics.

Provides a fast way to understand file structure, line counts,
import statements, function/class definitions, and key statistics
without reading the entire file.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "file_summary",
        "description": (
            "Generate a quick summary of a file's contents and structure. "
            "Returns statistics like line count, function/class definitions, "
            "import statements, and key patterns. Useful for quickly understanding "
            "what a file contains without reading it entirely. "
            "Supports Python, JavaScript, TypeScript, and text files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to summarize.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of content lines to include in preview (default: 50).",
                    "default": 50,
                },
            },
            "required": ["path"],
        },
    }


def _analyze_python_file(content: str, tree: ast.AST | None) -> dict[str, Any]:
    """Extract Python-specific information."""
    info = {
        "imports": [],
        "functions": [],
        "classes": [],
        "docstrings": [],
    }
    
    if tree is None:
        return info
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            info["imports"].append(f"{module}: {', '.join(names)}")
        elif isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "args": len(node.args.args),
                "line": node.lineno,
            }
            if ast.get_docstring(node):
                func_info["has_docstring"] = True
            info["functions"].append(func_info)
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                "line": node.lineno,
            }
            if ast.get_docstring(node):
                class_info["has_docstring"] = True
            info["classes"].append(class_info)
    
    return info


def _analyze_js_ts_file(content: str) -> dict[str, Any]:
    """Extract JavaScript/TypeScript-specific information."""
    info = {
        "imports": [],
        "exports": [],
        "functions": [],
        "classes": [],
    }
    
    # Find imports
    import_patterns = [
        r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
        r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        r'import\s+[\'"]([^\'"]+)[\'"]',
    ]
    for pattern in import_patterns:
        info["imports"].extend(re.findall(pattern, content))
    
    # Find exports
    export_pattern = r'export\s+(?:default\s+)?(?:class|function|const|let|var)?\s*(\w+)'
    info["exports"] = re.findall(export_pattern, content)
    
    # Find function definitions
    func_pattern = r'(?:function|const|let|var)\s+(\w+)\s*[=\(]'
    info["functions"] = re.findall(func_pattern, content)
    
    # Find class definitions
    class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
    classes = re.findall(class_pattern, content)
    info["classes"] = [{"name": c[0], "extends": c[1] if c[1] else None} for c in classes]
    
    return info


def _get_file_stats(file_path: Path, content: str) -> dict[str, Any]:
    """Get general file statistics."""
    lines = content.split('\n')
    
    stats = {
        "total_lines": len(lines),
        "non_empty_lines": len([l for l in lines if l.strip()]),
        "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "extension": file_path.suffix,
    }
    
    # Count comment lines (rough estimate)
    comment_patterns = {
        '.py': r'^\s*#',
        '.js': r'^\s*//',
        '.ts': r'^\s*//',
        '.jsx': r'^\s*//',
        '.tsx': r'^\s*//',
    }
    
    pattern = comment_patterns.get(file_path.suffix, r'^\s*#')
    stats["comment_lines"] = len([l for l in lines if re.match(pattern, l)])
    
    return stats


def tool_function(path: str, max_lines: int = 50) -> str:
    """Generate a summary of a file.

    Args:
        path: Absolute path to the file
        max_lines: Maximum content lines to include in preview

    Returns:
        Formatted summary of the file
    """
    file_path = Path(path)
    
    if not file_path.exists():
        return f"Error: File not found: {path}"
    
    if not file_path.is_file():
        return f"Error: Path is not a file: {path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"
    
    # Get file statistics
    stats = _get_file_stats(file_path, content)
    
    # Build summary
    summary_parts = [
        f"File: {file_path}",
        f"Type: {stats['extension'] or 'unknown'}",
        f"Size: {stats['file_size_bytes']:,} bytes",
        f"Lines: {stats['total_lines']:,} total, {stats['non_empty_lines']:,} non-empty, {stats['comment_lines']:,} comments",
        "",
    ]
    
    # Language-specific analysis
    suffix = file_path.suffix.lower()
    
    if suffix == '.py':
        try:
            tree = ast.parse(content)
            py_info = _analyze_python_file(content, tree)
            
            if py_info["imports"]:
                summary_parts.append(f"Imports ({len(py_info['imports'])}):")
                for imp in py_info["imports"][:10]:  # Limit to first 10
                    summary_parts.append(f"  - {imp}")
                if len(py_info["imports"]) > 10:
                    summary_parts.append(f"  ... and {len(py_info['imports']) - 10} more")
                summary_parts.append("")
            
            if py_info["classes"]:
                summary_parts.append(f"Classes ({len(py_info['classes'])}):")
                for cls in py_info["classes"]:
                    doc = " (has docstring)" if cls.get("has_docstring") else ""
                    summary_parts.append(f"  - {cls['name']} (line {cls['line']}, {cls['methods']} methods){doc}")
                summary_parts.append("")
            
            if py_info["functions"]:
                summary_parts.append(f"Functions ({len(py_info['functions'])}):")
                for func in py_info["functions"][:15]:  # Limit to first 15
                    doc = " (has docstring)" if func.get("has_docstring") else ""
                    summary_parts.append(f"  - {func['name']}({func['args']} args, line {func['line']}){doc}")
                if len(py_info["functions"]) > 15:
                    summary_parts.append(f"  ... and {len(py_info['functions']) - 15} more")
                summary_parts.append("")
                
        except SyntaxError:
            summary_parts.append("Note: File has Python syntax errors, skipping code analysis.")
            summary_parts.append("")
    
    elif suffix in ('.js', '.ts', '.jsx', '.tsx'):
        js_info = _analyze_js_ts_file(content)
        
        if js_info["imports"]:
            summary_parts.append(f"Imports ({len(js_info['imports'])}):")
            for imp in js_info["imports"][:10]:
                summary_parts.append(f"  - {imp}")
            if len(js_info["imports"]) > 10:
                summary_parts.append(f"  ... and {len(js_info['imports']) - 10} more")
            summary_parts.append("")
        
        if js_info["classes"]:
            summary_parts.append(f"Classes ({len(js_info['classes'])}):")
            for cls in js_info["classes"]:
                extends = f" extends {cls['extends']}" if cls['extends'] else ""
                summary_parts.append(f"  - {cls['name']}{extends}")
            summary_parts.append("")
        
        if js_info["exports"]:
            summary_parts.append(f"Exports ({len(js_info['exports'])}): {', '.join(js_info['exports'][:10])}")
            if len(js_info["exports"]) > 10:
                summary_parts.append(f"  ... and {len(js_info['exports']) - 10} more")
            summary_parts.append("")
    
    # Add content preview
    lines = content.split('\n')
    preview_lines = lines[:max_lines]
    
    summary_parts.append(f"Preview (first {len(preview_lines)} lines):")
    summary_parts.append("```")
    for i, line in enumerate(preview_lines, 1):
        # Truncate very long lines
        if len(line) > 100:
            line = line[:97] + "..."
        summary_parts.append(f"{i:4d}: {line}")
    if len(lines) > max_lines:
        summary_parts.append(f"... ({len(lines) - max_lines} more lines)")
    summary_parts.append("```")
    
    return "\n".join(summary_parts)


if __name__ == "__main__":
    # Test the tool
    import tempfile
    
    # Create a test Python file
    test_code = '''
"""Test module for demonstration."""

import os
import sys
from typing import List, Dict

class TestClass:
    """A test class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return f"Hello, {self.name}!"

def helper_function(x: int) -> int:
    """A helper function."""
    return x * 2

def main():
    obj = TestClass("World")
    print(obj.greet())
    print(helper_function(5))

if __name__ == "__main__":
    main()
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = f.name
    
    print(tool_function(temp_path))
    os.unlink(temp_path)
