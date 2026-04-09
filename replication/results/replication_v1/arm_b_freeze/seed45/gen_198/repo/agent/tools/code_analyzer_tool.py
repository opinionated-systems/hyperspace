"""
Code analyzer tool: analyze Python code for common issues.

Provides static analysis capabilities to identify:
- Unused imports
- Undefined variables
- Syntax errors
- Basic style issues
- Complexity metrics

This helps the meta agent identify code quality issues more effectively.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "code_analyzer",
        "description": (
            "Analyze Python code for common issues. "
            "Detects unused imports, undefined variables, syntax errors, "
            "and provides complexity metrics. "
            "Helps identify code quality issues and potential bugs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file or directory to analyze.",
                },
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["imports", "variables", "syntax", "complexity", "all"],
                    },
                    "description": "Types of checks to run (default: all).",
                },
                "max_complexity": {
                    "type": "integer",
                    "description": "Maximum acceptable cyclomatic complexity (default: 10).",
                    "default": 10,
                },
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict code analysis operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def tool_function(
    path: str,
    checks: list[str] | None = None,
    max_complexity: int = 10,
) -> str:
    """Analyze Python code for common issues."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        # Default to all checks if not specified
        if checks is None or "all" in checks:
            checks = ["imports", "variables", "syntax", "complexity"]
        
        results = []
        
        if p.is_file():
            if not str(p).endswith('.py'):
                return f"Error: {p} is not a Python file."
            file_results = _analyze_file(p, checks, max_complexity)
            results.append(file_results)
        else:
            # Analyze all Python files in directory
            py_files = list(p.rglob("*.py"))
            py_files = [f for f in py_files if not f.name.startswith('.') and '__pycache__' not in str(f)]
            
            if not py_files:
                return f"No Python files found in {p}"
            
            for f in py_files:
                file_results = _analyze_file(f, checks, max_complexity)
                results.append(file_results)
        
        return _format_results(results, str(p))
        
    except Exception as e:
        return f"Error: {e}"


def _analyze_file(path: Path, checks: list[str], max_complexity: int) -> dict[str, Any]:
    """Analyze a single Python file."""
    result = {
        "path": str(path),
        "issues": [],
        "metrics": {},
    }
    
    try:
        content = path.read_text(encoding='utf-8')
    except Exception as e:
        result["issues"].append(f"Could not read file: {e}")
        return result
    
    # Syntax check
    tree = None
    if "syntax" in checks:
        try:
            tree = ast.parse(content)
            result["metrics"]["lines"] = len(content.splitlines())
            result["metrics"]["functions"] = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            result["metrics"]["classes"] = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        except SyntaxError as e:
            result["issues"].append(f"Syntax error at line {e.lineno}: {e.msg}")
            return result
    else:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            result["issues"].append("File has syntax errors (could not parse)")
            return result
    
    # Import analysis
    if "imports" in checks:
        import_issues = _check_imports(tree, content)
        result["issues"].extend(import_issues)
    
    # Variable analysis
    if "variables" in checks:
        var_issues = _check_variables(tree)
        result["issues"].extend(var_issues)
    
    # Complexity analysis
    if "complexity" in checks:
        complexity_issues = _check_complexity(tree, max_complexity)
        result["issues"].extend(complexity_issues)
    
    return result


def _check_imports(tree: ast.AST, content: str) -> list[str]:
    """Check for unused imports."""
    issues = []
    
    # Find all imports
    imports = {}
    import_aliases = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                full_name = f"{module}.{alias.name}" if module else alias.name
                imports[name] = node.lineno
                import_aliases[name] = full_name
    
    # Find all name usages
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Handle module.attribute usage
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)
    
    # Check for unused imports
    # Skip __future__ imports (always used for type annotations)
    future_imports = {'annotations', 'division', 'print_function', 'unicode_literals',
                       'absolute_import', 'with_statement', 'nested_scopes', 'generators'}
    
    for name, lineno in imports.items():
        if name not in used_names and not name.startswith('_'):
            # Skip __future__ imports
            if name in future_imports:
                continue
            issues.append(f"Line {lineno}: Unused import '{name}'")
    
    return issues


def _check_variables(tree: ast.AST) -> list[str]:
    """Check for undefined variables and unused variables."""
    issues = []
    
    # Collect defined and used names per scope
    defined_names = set()
    used_names = set()
    
    # Builtins and common module-level names
    builtins = {
        'True', 'False', 'None', 'print', 'len', 'range', 'enumerate',
        'zip', 'map', 'filter', 'sum', 'min', 'max', 'abs', 'round',
        'int', 'float', 'str', 'list', 'dict', 'set', 'tuple', 'bool',
        'open', 'isinstance', 'hasattr', 'getattr', 'setattr', 'type',
        'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
        'AttributeError', 'RuntimeError', 'ImportError', 'ModuleNotFoundError',
        'SyntaxError', 'NameError', 'ZeroDivisionError', 'OverflowError',
        'FloatingPointError', 'OSError', 'IOError', 'FileNotFoundError',
        'PermissionError', 'NotADirectoryError', 'IsADirectoryError',
        'EOFError', 'MemoryError', 'RecursionError', 'SystemError',
        'ReferenceError', 'NotImplementedError', 'StopIteration',
        'StopAsyncIteration', 'GeneratorExit', 'SystemExit', 'KeyboardInterrupt',
        'LookupError', 'ArithmeticError', 'BufferError', 'AssertionError',
        'BlockingIOError', 'ChildProcessError', 'ConnectionError',
        'BrokenPipeError', 'ConnectionAbortedError', 'ConnectionRefusedError',
        'ConnectionResetError', 'InterruptedError', 'TimeoutError',
        'Warning', 'UserWarning', 'DeprecationWarning', 'PendingDeprecationWarning',
        'SyntaxWarning', 'RuntimeWarning', 'FutureWarning', 'ImportWarning',
        'UnicodeWarning', 'BytesWarning', 'ResourceWarning', 'EncodingWarning',
        'os', 'sys', 'json', 're', 'logging', 'pathlib', 'typing',
        'functools', 'itertools', 'collections', 'datetime', 'time',
        'math', 'random', 'string', 'hashlib', 'base64', 'urllib',
        'http', 'socket', 'subprocess', 'threading', 'multiprocessing',
        'unittest', 'pytest', 'ast', 'inspect', 'textwrap', 'copy',
        'pickle', 'csv', 'io', 'warnings', 'contextlib', 'dataclasses',
        'enum', 'abc', 'numbers', 'decimal', 'fractions', 'statistics',
        'html', 'xml', 'json', 'csv', 'sqlite3', 'uuid', 'secrets',
        'hmac', 'tempfile', 'shutil', 'glob', 'fnmatch', 'filecmp',
        'linecache', 'traceback', 'sysconfig', 'platform', 'ctypes',
        'weakref', 'types', 'pprint', 'reprlib', 'string', 're',
        'difflib', 'textwrap', 'unicodedata', 'stringprep', 'readline',
        'rlcompleter', 'code', 'codeop', 'site', 'builtins', '__builtins__',
        'Any', 'Dict', 'List', 'Optional', 'Tuple', 'Union', 'Callable',
        'Set', 'FrozenSet', 'Type', 'ClassVar', 'Final', 'Literal',
        'Protocol', 'runtime_checkable', 'TypedDict', 'NamedTuple',
        'NewType', 'NoReturn', 'Never', 'assert_never', 'Self',
        'Required', 'NotRequired', 'Unpack', 'TypedDict', 'Annotated',
        'get_type_hints', 'get_origin', 'get_args', 'TYPE_CHECKING',
        'Path', 'PurePath', 'PurePosixPath', 'PureWindowsPath', 'PosixPath', 'WindowsPath',
    }
    
    defined_names.update(builtins)
    
    # Walk the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            # Add function arguments as defined
            for arg in node.args.args:
                defined_names.add(arg.arg)
            for arg in node.args.kwonlyargs:
                defined_names.add(arg.arg)
            if node.args.vararg:
                defined_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined_names.add(node.args.kwarg.arg)
            # Add function name as defined
            defined_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            for arg in node.args.args:
                defined_names.add(arg.arg)
            defined_names.add(node.name)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined_names.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        defined_names.add(elt.id)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    if isinstance(item.optional_vars, ast.Name):
                        defined_names.add(item.optional_vars.id)
                    elif isinstance(item.optional_vars, ast.Tuple):
                        for elt in item.optional_vars.elts:
                            if isinstance(elt, ast.Name):
                                defined_names.add(elt.id)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined_names.add(node.name)
        elif isinstance(node, ast.Global):
            for name in node.names:
                defined_names.add(name)
        elif isinstance(node, ast.Nonlocal):
            for name in node.names:
                defined_names.add(name)
    
    # Check for undefined names
    undefined = used_names - defined_names
    for name in undefined:
        if not name.startswith('_') and name not in ['__name__', '__file__', '__doc__', '__package__', '__spec__', '__annotations__', '__builtins__', '__cached__', '__loader__']:
            # Find where it's used
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                    issues.append(f"Line {node.lineno}: Undefined name '{name}'")
                    break
    
    return issues


def _check_complexity(tree: ast.AST, max_complexity: int) -> list[str]:
    """Check cyclomatic complexity of functions."""
    issues = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = _calculate_complexity(node)
            if complexity > max_complexity:
                issues.append(f"Line {node.lineno}: Function '{node.name}' has complexity {complexity} (max: {max_complexity})")
    
    return issues


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of a function."""
    complexity = 1  # Base complexity
    
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.With):
            complexity += 1
        elif isinstance(child, ast.Assert):
            complexity += 1
    
    return complexity


def _format_results(results: list[dict], path: str) -> str:
    """Format analysis results for display."""
    lines = [f"Code Analysis Results for: {path}", "=" * 60]
    
    total_issues = 0
    total_files = len(results)
    files_with_issues = 0
    
    for result in results:
        file_path = result["path"]
        issues = result["issues"]
        metrics = result.get("metrics", {})
        
        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            lines.append(f"\n{file_path}:")
            lines.append("-" * 40)
            for issue in issues:
                lines.append(f"  • {issue}")
        
        if metrics:
            lines.append(f"\n  Metrics:")
            for key, value in metrics.items():
                lines.append(f"    - {key}: {value}")
    
    lines.append("\n" + "=" * 60)
    lines.append(f"Summary: {total_issues} issues found in {files_with_issues}/{total_files} files")
    
    if total_issues == 0:
        lines.append("✓ No issues found!")
    
    return "\n".join(lines)
