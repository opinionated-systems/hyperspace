"""
Validate tool: check Python code syntax and basic structure.

Provides early error detection for code modifications.
"""

from __future__ import annotations

import ast
import logging
import re

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "validate",
        "description": "Validate Python code syntax and structure. Checks for syntax errors, undefined variables, and common issues. Returns a report of any problems found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to validate",
                },
                "check_undefined": {
                    "type": "boolean",
                    "description": "Whether to check for potentially undefined variables (default: true)",
                },
            },
            "required": ["code"],
        },
    }


def tool_function(code: str, check_undefined: bool = True) -> str:
    """Validate Python code and return a report.
    
    Args:
        code: Python code to validate
        check_undefined: Whether to check for undefined variables
        
    Returns:
        Validation report as a string
    """
    if not code or not code.strip():
        return "Error: No code provided"
    
    issues = []
    
    # Check 1: Syntax validation using AST
    try:
        tree = ast.parse(code)
        issues.append("✓ Syntax is valid")
    except SyntaxError as e:
        issues.append(f"✗ Syntax Error: {e.msg} (line {e.lineno}, col {e.offset})")
        return "\n".join(issues)
    except Exception as e:
        issues.append(f"✗ Parse Error: {type(e).__name__}: {e}")
        return "\n".join(issues)
    
    # Check 2: Basic structure analysis
    try:
        # Count different node types
        function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        import_count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        
        issues.append(f"✓ Structure: {function_count} function(s), {class_count} class(es), {import_count} import(s)")
        
        # Check for empty functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.body or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                    issues.append(f"⚠ Function '{node.name}' appears to be empty or only contains 'pass'")
        
        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(f"⚠ Bare 'except:' clause found (line {node.lineno}) - consider using 'except Exception:'")
        
        # Check for mutable default arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults + node.args.kw_defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(f"⚠ Mutable default argument in function '{node.name}' (line {node.lineno}) - use None and initialize in body")
        
    except Exception as e:
        issues.append(f"⚠ Structure analysis failed: {e}")
    
    # Check 3: Common issues
    try:
        # Check for print statements (might be debug code)
        if re.search(r'\bprint\s*\(', code):
            issues.append("ℹ Found 'print()' calls - consider using logging instead")
        
        # Check for TODO/FIXME comments
        todo_matches = re.findall(r'#\s*(TODO|FIXME|XXX|HACK)', code, re.IGNORECASE)
        if todo_matches:
            issues.append(f"ℹ Found {len(todo_matches)} TODO/FIXME comment(s)")
        
        # Check for very long lines
        long_lines = [i+1 for i, line in enumerate(code.split('\n')) if len(line) > 120]
        if long_lines:
            issues.append(f"ℹ Very long lines (>120 chars) on: {', '.join(map(str, long_lines[:5]))}{'...' if len(long_lines) > 5 else ''}")
        
    except Exception as e:
        issues.append(f"⚠ Common issues check failed: {e}")
    
    # Check 4: Undefined variable detection (basic)
    if check_undefined:
        try:
            undefined = _check_undefined_vars(tree, code)
            if undefined:
                issues.append(f"⚠ Potentially undefined variables: {', '.join(undefined[:5])}{'...' if len(undefined) > 5 else ''}")
            else:
                issues.append("✓ No obvious undefined variables detected")
        except Exception as e:
            issues.append(f"⚠ Undefined variable check failed: {e}")
    
    return "\n".join(issues)


def _check_undefined_vars(tree: ast.AST, code: str) -> list[str]:
    """Basic check for potentially undefined variables.
    
    This is a simple heuristic and may have false positives.
    """
    defined = set()
    undefined = set()
    
    # Builtins and common imports
    builtins = {
        'True', 'False', 'None', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'sum', 'min', 'max', 'abs', 'round', 'int', 'float', 'str', 'list', 'dict',
        'set', 'tuple', 'bool', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
        'open', 'print', 'input', 'super', 'object', 'Exception', 'ValueError',
        'TypeError', 'KeyError', 'IndexError', 'AttributeError', 'RuntimeError',
        'ImportError', 'ModuleNotFoundError', 'StopIteration', 'GeneratorExit',
        'SystemExit', 'KeyboardInterrupt', 'ArithmeticError', 'LookupError',
        'AssertionError', 'BufferError', 'EOFError', 'FloatingPointError',
        'OSError', 'IOError', 'EnvironmentError', 'BlockingIOError',
        'ChildProcessError', 'ConnectionError', 'BrokenPipeError',
        'ConnectionAbortedError', 'ConnectionRefusedError', 'ConnectionResetError',
        'FileExistsError', 'FileNotFoundError', 'InterruptedError',
        'IsADirectoryError', 'NotADirectoryError', 'PermissionError',
        'ProcessLookupError', 'TimeoutError', 'ReferenceError', 'SyntaxError',
        'IndentationError', 'TabError', 'SystemError', 'UnicodeError',
        'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeTranslateError',
        'Warning', 'UserWarning', 'DeprecationWarning', 'PendingDeprecationWarning',
        'SyntaxWarning', 'RuntimeWarning', 'FutureWarning', 'ImportWarning',
        'UnicodeWarning', 'BytesWarning', 'ResourceWarning', 'NotImplementedError',
        'RecursionError', 'json', 're', 'os', 'sys', 'time', 'datetime', 'logging',
        'typing', 'collections', 'itertools', 'functools', 'math', 'random',
        'string', 'hashlib', 'base64', 'urllib', 'http', 'socket', 'threading',
        'multiprocessing', 'subprocess', 'pathlib', 'inspect', 'textwrap',
        'copy', 'pickle', 'csv', 'io', 'warnings', 'contextlib', 'dataclasses',
        'enum', 'abc', 'numbers', 'decimal', 'fractions', 'statistics',
        'uuid', 'html', 'xml', 'json', 'csv', 'sqlite3', 'zlib', 'gzip',
        'bz2', 'lzma', 'zipfile', 'tarfile', 'shutil', 'tempfile', 'glob',
        'fnmatch', 'linecache', 'traceback', 'sysconfig', 'site', 'importlib',
        'pkgutil', 'modulefinder', 'runpy', 'parser', 'ast', 'dis', 'pickletools',
        'code', 'codeop', 'pprint', 'reprlib', 'string', 're', 'difflib',
        'textwrap', 'unicodedata', 'stringprep', 'readline', 'rlcompleter',
        'struct', 'codecs', 'datetime', 'calendar', 'collections', 'heapq',
        'bisect', 'array', 'weakref', 'types', 'copy', 'pprint', 'reprlib',
        'enum', 'graphlib', 'numbers', 'math', 'cmath', 'decimal', 'fractions',
        'random', 'statistics', 'itertools', 'functools', 'operator',
        'pathlib', 'os', 'io', 'pickle', 'copyreg', 'shelve', 'marshal',
        'dbm', 'sqlite3', 'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
        'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib', 'hashlib',
        'hmac', 'secrets', 'os', 'io', 'time', 'argparse', 'getopt', 'logging',
        'getpass', 'curses', 'platform', 'errno', 'ctypes', 'threading',
        'multiprocessing', 'concurrent', 'subprocess', 'sched', 'queue',
        'contextvars', 'asyncio', 'socket', 'ssl', 'select', 'selectors',
        'asyncore', 'asynchat', 'signal', 'mmap', 'email', 'json', 'mailbox',
        'mimetypes', 'base64', 'binhex', 'binascii', 'quopri', 'uu',
        'html', 'xml', 'webbrowser', 'cgi', 'cgitb', 'wsgiref', 'urllib',
        'http', 'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib',
        'smtpd', 'telnetlib', 'uuid', 'socketserver', 'http', 'xmlrpc',
        'ipaddress', 'audioop', 'aifc', 'sunau', 'wave', 'chunk', 'colorsys',
        'imghdr', 'sndhdr', 'ossaudiodev', 'spwd', 'crypt', 'msilib', 'nis',
        'nntplib', 'optparse', 'ossaudiodev', 'spwd', 'sunau', 'telnetlib',
        'uu', 'xdrlib', '__future__', 'annotations', 'Any', 'Callable',
        'Dict', 'List', 'Optional', 'Set', 'Tuple', 'Union', 'TypeVar',
        'Generic', 'Protocol', 'runtime_checkable', 'Final', 'Literal',
        'TypedDict', 'NamedTuple', 'NewType', 'NoReturn', 'ClassVar',
        'Pattern', 'Match', 'Match', 'Pattern', 'Match', 'Pattern', 'Match',
    }
    defined.update(builtins)
    
    for node in ast.walk(tree):
        # Track assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        defined.add(elt.id)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    if isinstance(item.optional_vars, ast.Name):
                        defined.add(item.optional_vars.id)
                    elif isinstance(item.optional_vars, ast.Tuple):
                        for elt in item.optional_vars.elts:
                            if isinstance(elt, ast.Name):
                                defined.add(elt.id)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            # Add parameters
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                defined.add(arg.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split('.')[0]
                defined.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                defined.add(name)
        elif isinstance(node, ast.Global):
            for name in node.names:
                defined.add(name)
        elif isinstance(node, ast.Nonlocal):
            for name in node.names:
                defined.add(name)
        
        # Check for name usage
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                if node.id not in defined:
                    undefined.add(node.id)
    
    return sorted(undefined)
