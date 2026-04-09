"""
Code analysis tool: analyze Python code structure, imports, classes, and functions.

Provides static analysis capabilities to help the agent understand code structure
before making modifications. This improves the quality and safety of code changes.
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
            "Analyze Python code structure to understand imports, classes, functions, "
            "and their relationships. Helps make informed code modifications. "
            "Commands: analyze_file, analyze_directory, find_symbol, get_dependencies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["analyze_file", "analyze_directory", "find_symbol", "get_dependencies"],
                    "description": "The analysis command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory to analyze.",
                },
                "symbol": {
                    "type": "string",
                    "description": "Symbol name to find (for find_symbol command).",
                },
                "include_private": {
                    "type": "boolean",
                    "description": "Include private members (starting with _) in analysis.",
                    "default": False,
                },
            },
            "required": ["command", "path"],
        },
    }


class CodeAnalyzer:
    """Analyzes Python code structure using AST."""

    def __init__(self, include_private: bool = False):
        self.include_private = include_private

    def analyze_file(self, filepath: Path) -> dict[str, Any]:
        """Analyze a single Python file."""
        if not filepath.exists():
            return {"error": f"File {filepath} does not exist"}
        
        if not filepath.suffix == ".py":
            return {"error": f"File {filepath} is not a Python file"}

        try:
            content = filepath.read_text()
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"Syntax error in {filepath}: {e}"}
        except Exception as e:
            return {"error": f"Failed to parse {filepath}: {e}"}

        result = {
            "file": str(filepath),
            "imports": [],
            "classes": [],
            "functions": [],
            "docstring": ast.get_docstring(tree),
            "line_count": len(content.splitlines()),
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append({
                        "type": "import",
                        "name": alias.name,
                        "asname": alias.asname,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    result["imports"].append({
                        "type": "from",
                        "module": module,
                        "name": alias.name,
                        "asname": alias.asname,
                    })
            elif isinstance(node, ast.ClassDef):
                if not self.include_private and node.name.startswith("_"):
                    continue
                class_info = self._analyze_class(node)
                result["classes"].append(class_info)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if not self.include_private and node.name.startswith("_"):
                    continue
                # Skip methods, they're handled in class analysis
                if not isinstance(node.parent, ast.ClassDef) if hasattr(node, "parent") else True:
                    func_info = self._analyze_function(node)
                    result["functions"].append(func_info)

        # Add parent references for proper method detection
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                setattr(child, "parent", node)

        return result

    def _analyze_class(self, node: ast.ClassDef) -> dict[str, Any]:
        """Analyze a class definition."""
        bases = [self._get_name(base) for base in node.bases]
        
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                if not self.include_private and item.name.startswith("_"):
                    continue
                methods.append(self._analyze_function(item))
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if not self.include_private and item.target.id.startswith("_"):
                    continue
                attributes.append({
                    "name": item.target.id,
                    "annotation": self._get_annotation(item.annotation) if item.annotation else None,
                })

        return {
            "name": node.name,
            "bases": bases,
            "docstring": ast.get_docstring(node),
            "methods": methods,
            "attributes": attributes,
            "line_number": node.lineno,
        }

    def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
        """Analyze a function definition."""
        args = []
        for arg in node.args.args:
            arg_info = {"name": arg.arg}
            if arg.annotation:
                arg_info["annotation"] = self._get_annotation(arg.annotation)
            args.append(arg_info)

        # Add *args and **kwargs
        if node.args.vararg:
            args.append({"name": f"*{node.args.vararg.arg}"})
        if node.args.kwarg:
            args.append({"name": f"**{node.args.kwarg.arg}"})

        returns = None
        if node.returns:
            returns = self._get_annotation(node.returns)

        return {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": args,
            "returns": returns,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "line_number": node.lineno,
        }

    def _get_name(self, node: ast.AST) -> str:
        """Get name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[...]"
        return str(type(node).__name__)

    def _get_annotation(self, node: ast.AST) -> str:
        """Get string representation of a type annotation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation(node.value)
            slice_val = self._get_annotation(node.slice) if hasattr(node, "slice") else "..."
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._get_annotation(node.left)
            right = self._get_annotation(node.right)
            return f"{left} | {right}"
        return "..."

    def analyze_directory(self, dirpath: Path) -> dict[str, Any]:
        """Analyze all Python files in a directory."""
        if not dirpath.exists():
            return {"error": f"Directory {dirpath} does not exist"}
        
        if not dirpath.is_dir():
            return {"error": f"{dirpath} is not a directory"}

        results = {
            "directory": str(dirpath),
            "files": [],
            "total_classes": 0,
            "total_functions": 0,
            "total_lines": 0,
        }

        for root, _, files in os.walk(dirpath):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = Path(root) / filename
                    file_analysis = self.analyze_file(filepath)
                    if "error" not in file_analysis:
                        results["files"].append(file_analysis)
                        results["total_classes"] += len(file_analysis.get("classes", []))
                        results["total_functions"] += len(file_analysis.get("functions", []))
                        results["total_lines"] += file_analysis.get("line_count", 0)

        return results

    def find_symbol(self, dirpath: Path, symbol: str) -> list[dict[str, Any]]:
        """Find a symbol (class or function) across files."""
        results = []
        
        for root, _, files in os.walk(dirpath):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = Path(root) / filename
                    file_analysis = self.analyze_file(filepath)
                    
                    if "error" in file_analysis:
                        continue

                    # Check classes
                    for cls in file_analysis.get("classes", []):
                        if cls["name"] == symbol:
                            results.append({
                                "type": "class",
                                "file": str(filepath),
                                "line": cls["line_number"],
                                "info": cls,
                            })

                    # Check functions
                    for func in file_analysis.get("functions", []):
                        if func["name"] == symbol:
                            results.append({
                                "type": "function",
                                "file": str(filepath),
                                "line": func["line_number"],
                                "info": func,
                            })

        return results

    def get_dependencies(self, filepath: Path) -> dict[str, Any]:
        """Get import dependencies for a file."""
        analysis = self.analyze_file(filepath)
        
        if "error" in analysis:
            return analysis

        stdlib = []
        third_party = []
        local = []

        stdlib_modules = {
            "os", "sys", "pathlib", "typing", "json", "re", "time", "datetime",
            "collections", "itertools", "functools", "hashlib", "logging",
            "threading", "concurrent", "subprocess", "ast", "inspect", "importlib",
        }

        for imp in analysis.get("imports", []):
            if imp["type"] == "import":
                module = imp["name"].split(".")[0]
            else:
                module = imp.get("module", "").split(".")[0]

            if module in stdlib_modules:
                stdlib.append(imp)
            elif module.startswith(".") or module == "":
                local.append(imp)
            else:
                third_party.append(imp)

        return {
            "file": str(filepath),
            "stdlib": stdlib,
            "third_party": third_party,
            "local": local,
        }


def _format_analysis(result: dict[str, Any]) -> str:
    """Format analysis result as readable text."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = []
    
    if "file" in result:
        lines.append(f"Analysis of {result['file']}:")
        lines.append(f"  Lines: {result.get('line_count', 'N/A')}")
        
        if result.get("docstring"):
            lines.append(f"  Module docstring: {result['docstring'][:100]}...")

        if result.get("imports"):
            lines.append(f"\n  Imports ({len(result['imports'])}):")
            for imp in result["imports"][:10]:
                if imp["type"] == "import":
                    name = imp["name"]
                    if imp.get("asname"):
                        name += f" as {imp['asname']}"
                    lines.append(f"    import {name}")
                else:
                    name = imp["name"]
                    if imp.get("asname"):
                        name += f" as {imp['asname']}"
                    lines.append(f"    from {imp.get('module', '')} import {name}")
            if len(result["imports"]) > 10:
                lines.append(f"    ... and {len(result['imports']) - 10} more")

        if result.get("classes"):
            lines.append(f"\n  Classes ({len(result['classes'])}):")
            for cls in result["classes"]:
                bases = f"({', '.join(cls['bases'])})" if cls.get("bases") else ""
                lines.append(f"    class {cls['name']}{bases} - line {cls['line_number']}")
                if cls.get("methods"):
                    method_names = [m["name"] for m in cls["methods"]]
                    lines.append(f"      methods: {', '.join(method_names[:5])}")

        if result.get("functions"):
            lines.append(f"\n  Functions ({len(result['functions'])}):")
            for func in result["functions"]:
                args = [a["name"] for a in func.get("args", [])]
                async_prefix = "async " if func.get("is_async") else ""
                lines.append(f"    {async_prefix}def {func['name']}({', '.join(args)}) - line {func['line_number']}")

    elif "directory" in result:
        lines.append(f"Analysis of directory {result['directory']}:")
        lines.append(f"  Files: {len(result.get('files', []))}")
        lines.append(f"  Total classes: {result.get('total_classes', 0)}")
        lines.append(f"  Total functions: {result.get('total_functions', 0)}")
        lines.append(f"  Total lines: {result.get('total_lines', 0)}")

    return "\n".join(lines)


def _format_symbol_results(results: list[dict[str, Any]], symbol: str) -> str:
    """Format symbol search results."""
    if not results:
        return f"Symbol '{symbol}' not found."

    lines = [f"Found '{symbol}' in {len(results)} location(s):"]
    for r in results:
        lines.append(f"\n  {r['type'].upper()}: {r['file']}:{r['line']}")
        if r['type'] == 'class':
            info = r['info']
            if info.get('bases'):
                lines.append(f"    Inherits: {', '.join(info['bases'])}")
            if info.get('methods'):
                method_names = [m['name'] for m in info['methods']]
                lines.append(f"    Methods: {', '.join(method_names)}")
        elif r['type'] == 'function':
            info = r['info']
            args = [a['name'] for a in info.get('args', [])]
            lines.append(f"    Signature: {info['name']}({', '.join(args)})")

    return "\n".join(lines)


def _format_dependencies(result: dict[str, Any]) -> str:
    """Format dependency analysis."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = [f"Dependencies for {result['file']}:"]
    
    if result.get("stdlib"):
        lines.append(f"\n  Standard library ({len(result['stdlib'])}):")
        for imp in result["stdlib"][:5]:
            if imp["type"] == "import":
                lines.append(f"    import {imp['name']}")
            else:
                lines.append(f"    from {imp.get('module', '')} import {imp['name']}")

    if result.get("third_party"):
        lines.append(f"\n  Third-party ({len(result['third_party'])}):")
        for imp in result["third_party"][:5]:
            if imp["type"] == "import":
                lines.append(f"    import {imp['name']}")
            else:
                lines.append(f"    from {imp.get('module', '')} import {imp['name']}")

    if result.get("local"):
        lines.append(f"\n  Local imports ({len(result['local'])}):")
        for imp in result["local"][:5]:
            if imp["type"] == "import":
                lines.append(f"    import {imp['name']}")
            else:
                lines.append(f"    from {imp.get('module', '')} import {imp['name']}")

    return "\n".join(lines)


def tool_function(
    command: str,
    path: str,
    symbol: str | None = None,
    include_private: bool = False,
) -> str:
    """Execute a code analysis command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        analyzer = CodeAnalyzer(include_private=include_private)

        if command == "analyze_file":
            result = analyzer.analyze_file(p)
            return _format_analysis(result)

        elif command == "analyze_directory":
            result = analyzer.analyze_directory(p)
            return _format_analysis(result)

        elif command == "find_symbol":
            if not symbol:
                return "Error: symbol parameter required for find_symbol command."
            if not p.is_dir():
                return f"Error: {path} must be a directory for find_symbol."
            results = analyzer.find_symbol(p, symbol)
            return _format_symbol_results(results, symbol)

        elif command == "get_dependencies":
            result = analyzer.get_dependencies(p)
            return _format_dependencies(result)

        else:
            return f"Error: unknown command {command}"

    except Exception as e:
        return f"Error: {e}"
