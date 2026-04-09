"""
Code analyzer tool: analyze Python code for common issues and quality metrics.

Provides static analysis capabilities to detect:
- Syntax errors
- Unused imports
- Undefined variables
- Basic code complexity metrics
- Style issues

This helps the meta agent understand code quality and identify improvement opportunities.
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
            "Analyze Python code for common issues and quality metrics. "
            "Detects syntax errors, unused imports, undefined variables, "
            "and provides complexity metrics. Helps identify code improvement opportunities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file or directory to analyze.",
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Whether to include complexity metrics (default: true).",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to analyze in a directory (default: 20).",
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
            return False, f"Error: access denied. Analysis restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_list(items: list[str], max_items: int = 20) -> list[str]:
    """Truncate list to max_items."""
    if len(items) > max_items:
        return items[:max_items] + [f"... ({len(items) - max_items} more items)"]
    return items


def tool_function(
    path: str,
    include_metrics: bool = True,
    max_files: int = 20,
) -> str:
    """Analyze Python code at the given path."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if p.is_dir():
            return _analyze_directory(p, include_metrics, max_files)
        elif p.suffix == ".py":
            return _analyze_file(p, include_metrics)
        else:
            return f"Error: {p} is not a Python file or directory."
            
    except Exception as e:
        return f"Error: {e}"


def _analyze_directory(path: Path, include_metrics: bool, max_files: int) -> str:
    """Analyze all Python files in a directory."""
    py_files = list(path.rglob("*.py"))
    
    # Exclude common non-source directories
    exclude_patterns = [
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".pytest_cache", ".mypy_cache", ".tox", "build", "dist"
    ]
    
    def should_exclude(f: Path) -> bool:
        """Check if file path contains any exclude pattern."""
        f_str = str(f)
        return any(pattern in f_str for pattern in exclude_patterns)
    
    py_files = [f for f in py_files if not should_exclude(f)]
    
    if not py_files:
        return f"No Python files found in {path}"
    
    # Sort and limit
    py_files.sort()
    total_files = len(py_files)
    py_files = py_files[:max_files]
    
    results = []
    issues_found = 0
    files_with_issues = 0
    
    for file_path in py_files:
        file_result = _analyze_file_internal(file_path, include_metrics)
        if file_result["issues"]:
            issues_found += len(file_result["issues"])
            files_with_issues += 1
            results.append(f"\n{'='*60}")
            results.append(f"File: {file_path}")
            results.append(f"{'='*60}")
            for issue in file_result["issues"]:
                results.append(f"  • {issue}")
            if include_metrics and file_result["metrics"]:
                results.append(f"\n  Metrics:")
                for metric, value in file_result["metrics"].items():
                    results.append(f"    - {metric}: {value}")
    
    if not results:
        return f"✓ Analyzed {min(total_files, max_files)} Python files - no issues found!"
    
    summary = f"Code Analysis Summary for {path}:\n"
    summary += f"  Files analyzed: {len(py_files)} of {total_files}\n"
    summary += f"  Files with issues: {files_with_issues}\n"
    summary += f"  Total issues: {issues_found}\n"
    summary += "\n".join(results)
    
    return summary


def _analyze_file(path: Path, include_metrics: bool) -> str:
    """Analyze a single Python file."""
    result = _analyze_file_internal(path, include_metrics)
    
    output = [f"Code Analysis for {path}:", "=" * 60]
    
    if result["issues"]:
        output.append("\nIssues Found:")
        for issue in result["issues"]:
            output.append(f"  • {issue}")
    else:
        output.append("\n✓ No issues found!")
    
    if include_metrics and result["metrics"]:
        output.append("\nMetrics:")
        for metric, value in result["metrics"].items():
            output.append(f"  • {metric}: {value}")
    
    return "\n".join(output)


def _analyze_file_internal(path: Path, include_metrics: bool) -> dict[str, Any]:
    """Internal analysis function returning structured data."""
    issues = []
    metrics = {}
    
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"issues": [f"Error reading file: {e}"], "metrics": {}}
    
    # Check for syntax errors
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        issues.append(f"Syntax Error at line {e.lineno}: {e.msg}")
        return {"issues": issues, "metrics": {}}
    except Exception as e:
        issues.append(f"Parse Error: {e}")
        return {"issues": issues, "metrics": {}}
    
    # Analyze imports
    import_issues = _analyze_imports(tree, source)
    issues.extend(import_issues)
    
    # Analyze undefined variables
    undefined_issues = _analyze_undefined_vars(tree, source)
    issues.extend(undefined_issues)
    
    # Analyze common issues
    common_issues = _analyze_common_issues(tree, source)
    issues.extend(common_issues)
    
    # Calculate metrics if requested
    if include_metrics:
        metrics = _calculate_metrics(tree, source)
    
    return {"issues": _truncate_list(issues, 30), "metrics": metrics}


def _analyze_imports(tree: ast.AST, source: str) -> list[str]:
    """Check for unused imports."""
    issues = []
    
    # Collect all imports
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = {"node": node, "used": False, "line": node.lineno}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # Skip __future__ imports - they affect parsing and are always "used"
            if module == "__future__":
                continue
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = {"node": node, "used": False, "line": node.lineno}
    
    # Check for usage
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in imports:
                imports[node.id]["used"] = True
        elif isinstance(node, ast.Attribute):
            # Handle module.attribute usage (e.g., os.path, stat.S_IRUSR)
            if isinstance(node.value, ast.Name):
                if node.value.id in imports:
                    imports[node.value.id]["used"] = True
                # Also mark the attribute name as used if it's an import
                if node.attr in imports:
                    imports[node.attr]["used"] = True
    
    # Report unused imports
    for name, info in imports.items():
        if not info["used"]:
            issues.append(f"Unused import '{name}' at line {info['line']}")
    
    return issues


def _analyze_undefined_vars(tree: ast.AST, source: str) -> list[str]:
    """Check for potentially undefined variables."""
    issues = []
    
    # Track defined names per scope
    defined_names = [set()]  # Stack of scopes
    # Get builtin names - handle both module and dict cases
    try:
        if isinstance(__builtins__, dict):
            builtin_names = set(__builtins__.keys())
        else:
            builtin_names = set(dir(__builtins__))
    except:
        builtin_names = set(dir(__builtins__) if hasattr(__builtins__, '__class__') else [])
    
    # Add common type hints and typing constructs that might appear
    builtin_names.update([
        'annotations', 'Any', 'Dict', 'List', 'Set', 'Tuple', 'Optional', 'Union',
        'Callable', 'Type', 'ClassVar', 'Final', 'Literal', 'TypedDict',
        'Protocol', 'runtime_checkable', 'abstractmethod', 'final', 'overload',
        # Common module names that are typically imported
        'os', 'sys', 'json', 're', 'ast', 'pathlib', 'typing', 'subprocess',
        'threading', 'time', 'datetime', 'logging', 'collections', 'itertools',
        'functools', 'math', 'random', 'string', 'hashlib', 'base64', 'copy',
        'pickle', 'csv', 'io', 'tempfile', 'shutil', 'glob', 'fnmatch',
        'inspect', 'textwrap', 'warnings', 'contextlib', 'dataclasses',
        'enum', 'types', 'numbers', 'decimal', 'fractions', 'statistics',
        'html', 'xml', 'urllib', 'http', 'socket', 'email', 'uuid',
        'secrets', 'hmac', 'ssl', 'ftplib', 'smtplib', 'imaplib', 'poplib',
        'nntplib', 'telnetlib', 'webbrowser', 'cgi', 'wsgiref', 'urllib',
        'http', 'socketserver', 'http', 'xmlrpc', 'ipaddress', 'mailbox',
        'mimetypes', 'base64', 'binhex', 'binascii', 'quopri', 'uu',
        'html', 'xml', 'xmlrpc', 'cgi', 'wsgiref', 'http', 'ftplib',
        'smtplib', 'poplib', 'imaplib', 'nntplib', 'telnetlib', 'uuid',
        'socketserver', 'http', 'xmlrpc', 'ipaddress', 'mailbox',
        'mimetypes', 'base64', 'binhex', 'binascii', 'quopri', 'uu',
        # Special Python names
        '__builtins__', '__name__', '__file__', '__doc__', '__package__',
        '__spec__', '__loader__', '__cached__', '__annotations__',
        # Common imported types
        'Path', 'PurePath', 'PosixPath', 'WindowsPath'
    ])
    
    # Collect all function and class names defined at module level
    # These are available throughout the module
    module_level_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_level_names.add(node.name)
    builtin_names.update(module_level_names)
    
    # Also collect all imported names - they are defined at module level
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                # For "import os.path", add both "os" and the full module path
                if "." in name:
                    module_level_names.add(name.split(".")[0])
                module_level_names.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                module_level_names.add(name)
    builtin_names.update(module_level_names)
    
    class UndefinedVisitor(ast.NodeVisitor):
        def __init__(self):
            self.issues = []
            self.global_names = set()
            self.nonlocal_names = set()
        
        def visit_Global(self, node):
            for name in node.names:
                self.global_names.add(name)
            self.generic_visit(node)
        
        def visit_Nonlocal(self, node):
            for name in node.names:
                self.nonlocal_names.add(name)
            self.generic_visit(node)
        
        def visit_FunctionDef(self, node):
            # Add function name to outer scope
            if len(defined_names) > 0:
                defined_names[-1].add(node.name)
            # Create new scope
            defined_names.append(set())
            # Add parameters to new scope
            for arg in node.args.args:
                defined_names[-1].add(arg.arg)
            for arg in node.args.kwonlyargs:
                defined_names[-1].add(arg.arg)
            if node.args.vararg:
                defined_names[-1].add(node.args.vararg.arg)
            if node.args.kwarg:
                defined_names[-1].add(node.args.kwarg.arg)
            self.generic_visit(node)
            defined_names.pop()
        
        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)
        
        def visit_ClassDef(self, node):
            if len(defined_names) > 0:
                defined_names[-1].add(node.name)
            defined_names.append(set())
            self.generic_visit(node)
            defined_names.pop()
        
        def visit_Lambda(self, node):
            defined_names.append(set())
            for arg in node.args.args:
                defined_names[-1].add(arg.arg)
            self.generic_visit(node)
            defined_names.pop()
        
        def visit_ListComp(self, node):
            defined_names.append(set())
            self._handle_comprehension(node)
            self.generic_visit(node)
            defined_names.pop()
        
        def visit_SetComp(self, node):
            self.visit_ListComp(node)
        
        def visit_DictComp(self, node):
            defined_names.append(set())
            self._handle_comprehension(node)
            self.generic_visit(node)
            defined_names.pop()
        
        def visit_GeneratorExp(self, node):
            self.visit_ListComp(node)
        
        def _handle_comprehension(self, node):
            for generator in node.generators:
                self._add_target_names(generator.target)
        
        def _add_target_names(self, node):
            if isinstance(node, ast.Name):
                defined_names[-1].add(node.id)
            elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
                for elt in node.elts:
                    self._add_target_names(elt)
        
        def visit_For(self, node):
            self._add_target_names(node.target)
            self.generic_visit(node)
        
        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars:
                    self._add_target_names(item.optional_vars)
            self.generic_visit(node)
        
        def visit_ExceptHandler(self, node):
            if node.name:
                defined_names[-1].add(node.name)
            self.generic_visit(node)
        
        def visit_Assign(self, node):
            for target in node.targets:
                self._add_target_names(target)
            self.generic_visit(node)
        
        def visit_AnnAssign(self, node):
            if node.target:
                self._add_target_names(node.target)
            self.generic_visit(node)
        
        def visit_NamedExpr(self, node):
            defined_names[-1].add(node.target.id)
            self.generic_visit(node)
        
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                name = node.id
                if name not in builtin_names and name not in self.global_names and name not in self.nonlocal_names:
                    # Check if defined in any scope
                    if not any(name in scope for scope in defined_names):
                        self.issues.append(f"Potentially undefined variable '{name}' at line {node.lineno}")
            self.generic_visit(node)
    
    visitor = UndefinedVisitor()
    visitor.visit(tree)
    
    return visitor.issues[:20]  # Limit issues


def _analyze_common_issues(tree: ast.AST, source: str) -> list[str]:
    """Check for common code issues."""
    issues = []
    
    # Check for bare except clauses
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(
                    f"Bare 'except:' clause at line {node.lineno} "
                    "(catches all exceptions including KeyboardInterrupt)"
                )
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                issues.append(f"Overly broad 'except Exception:' at line {node.lineno}")
    
    # Check for mutable default arguments
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if default is None:
                    continue
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append(
                        f"Mutable default argument at line {node.lineno} "
                        "(use None and initialize in function body)"
                    )
                elif isinstance(default, ast.Call):
                    if isinstance(default.func, ast.Name) and default.func.id in ["list", "dict", "set"]:
                        issues.append(
                            f"Mutable default argument at line {node.lineno} "
                            "(use None and initialize in function body)"
                        )
    
    # Check for == vs is with None
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, ast.Eq):
                    # Check if comparing to None with ==
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Constant) and comparator.value is None:
                            issues.append(f"Using '== None' at line {node.lineno} (use 'is None' instead)")
                        elif isinstance(comparator, ast.NameConstant) and comparator.value is None:
                            issues.append(f"Using '== None' at line {node.lineno} (use 'is None' instead)")
    
    # Check for long lines
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            issues.append(f"Very long line ({len(line)} chars) at line {i}")
    
    return issues


def _calculate_metrics(tree: ast.AST, source: str) -> dict[str, Any]:
    """Calculate code complexity metrics."""
    metrics = {}
    
    lines = source.split("\n")
    
    # Basic metrics
    metrics["total_lines"] = len(lines)
    metrics["code_lines"] = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    metrics["blank_lines"] = len([l for l in lines if not l.strip()])
    metrics["comment_lines"] = len([l for l in lines if l.strip().startswith("#")])
    
    # Count definitions
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    
    metrics["functions"] = len(functions)
    metrics["classes"] = len(classes)
    
    # Calculate average function length
    if functions:
        func_lengths = []
        for func in functions:
            # Estimate function length by counting lines
            if hasattr(func, "end_lineno") and func.end_lineno:
                length = func.end_lineno - func.lineno + 1
                func_lengths.append(length)
        if func_lengths:
            metrics["avg_function_length"] = round(sum(func_lengths) / len(func_lengths), 1)
            metrics["max_function_length"] = max(func_lengths)
    
    # Cyclomatic complexity approximation
    complexity = 1  # Base complexity
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, ast.Compare):
            complexity += len(node.ops)
    
    metrics["cyclomatic_complexity"] = complexity
    
    # Complexity rating
    if complexity <= 10:
        metrics["complexity_rating"] = "Low (good)"
    elif complexity <= 20:
        metrics["complexity_rating"] = "Moderate"
    else:
        metrics["complexity_rating"] = "High (consider refactoring)"
    
    return metrics
