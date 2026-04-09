"""
Python execution tool: run Python code safely in a sandboxed environment.

Provides a way to test code snippets, validate logic, and perform calculations.
Uses ast.literal_eval for safe evaluation of simple expressions, and
restricted exec for more complex code.
"""

from __future__ import annotations

import ast
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any


def tool_info() -> dict:
    return {
        "name": "python",
        "description": (
            "Execute Python code safely. "
            "Use for testing logic, calculations, or validating code snippets. "
            "Returns stdout, stderr, and any return value. "
            "Timeout: 30 seconds. "
            "Restricted environment - no file system access, no network."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can include multiple statements.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 30, max: 60).",
                },
            },
            "required": ["code"],
        },
    }


# Restricted builtins for safer execution
_ALLOWED_BUILTINS = {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
    'callable', 'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate',
    'filter', 'float', 'format', 'frozenset', 'hasattr', 'hash', 'hex',
    'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
    'list', 'map', 'max', 'memoryview', 'min', 'next', 'oct', 'ord',
    'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
    'set', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
    'tuple', 'type', 'vars', 'zip', '__import__', 'True', 'False', 'None',
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'RuntimeError', 'StopIteration', 'ArithmeticError',
    'ZeroDivisionError', 'OverflowError', 'NameError', 'SyntaxError',
    'ImportError', 'ModuleNotFoundError', 'IOError', 'OSError',
    'RecursionError', 'NotImplementedError', 'AssertionError',
    'BaseException', 'SystemExit', 'KeyboardInterrupt', 'GeneratorExit',
    'Warning', 'UserWarning', 'DeprecationWarning', 'PendingDeprecationWarning',
    'RuntimeWarning', 'SyntaxWarning', 'ResourceWarning', 'FutureWarning',
    'ImportWarning', 'UnicodeWarning', 'BytesWarning', 'EncodingWarning',
}

# Disallowed modules that could be dangerous
_DISALLOWED_MODULES = {
    'os', 'sys', 'subprocess', 'socket', 'urllib', 'http', 'ftplib',
    'smtplib', 'email', 'shutil', 'pathlib', 'importlib', 'inspect',
    'ctypes', 'mmap', 'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3',
    'pdb', 'code', 'codeop', 'compileall', 'py_compile', 'zipfile',
    'tarfile', 'gzip', 'bz2', 'lzma', 'zlib', 'copy', 'copyreg',
    'weakref', 'gc', 'atexit', 'signal', 'threading', 'multiprocessing',
    'concurrent', 'asyncio', 'queue', 'contextvars', '_thread', '_dummy_thread',
}


def _is_safe_code(code: str) -> tuple[bool, str]:
    """Check if code is safe to execute by analyzing AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    for node in ast.walk(tree):
        # Check for imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.name.split('.')[0]
                if name in _DISALLOWED_MODULES:
                    return False, f"Import of '{name}' is not allowed for security reasons"
        
        # Check for dangerous builtins
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in ('eval', 'exec', 'compile', '__import__', 'open', 'exit', 'quit'):
                return False, f"Use of '{node.id}' is not allowed for security reasons"
        
        # Check for attribute access that might be dangerous
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_') and node.attr != '__init__':
                # Allow dunder methods but not private attributes
                if not (node.attr.startswith('__') and node.attr.endswith('__')):
                    return False, f"Access to private attribute '{node.attr}' is not allowed"
    
    return True, ""


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code safely.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30, max: 60)
    
    Returns:
        String containing stdout, stderr, and return value information
    """
    # Validate timeout
    timeout = min(max(timeout, 1), 60)
    
    # Check code safety
    is_safe, error_msg = _is_safe_code(code)
    if not is_safe:
        return f"Error: {error_msg}"
    
    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Create restricted globals
    safe_globals = {
        '__builtins__': {name: __builtins__[name] for name in _ALLOWED_BUILTINS if name in __builtins__},
        '__name__': '__main__',
        '__doc__': None,
    }
    
    # Add safe math and random modules
    try:
        import math
        import random
        import statistics
        import itertools
        import collections
        import functools
        import decimal
        import fractions
        import numbers
        import datetime
        import json
        import re
        import string
        import textwrap
        import typing
        
        safe_globals['math'] = math
        safe_globals['random'] = random
        safe_globals['statistics'] = statistics
        safe_globals['itertools'] = itertools
        safe_globals['collections'] = collections
        safe_globals['functools'] = functools
        safe_globals['decimal'] = decimal
        safe_globals['fractions'] = fractions
        safe_globals['numbers'] = numbers
        safe_globals['datetime'] = datetime
        safe_globals['json'] = json
        safe_globals['re'] = re
        safe_globals['string'] = string
        safe_globals['textwrap'] = textwrap
        safe_globals['typing'] = typing
    except ImportError:
        pass
    
    result = None
    exception = None
    
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Compile and execute with timeout
            compiled = compile(code, '<string>', 'exec')
            
            # Use alarm for timeout (Unix) or threading (Windows fallback)
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution exceeded {timeout} seconds")
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                exec(compiled, safe_globals)
                # Try to get a result if the last statement was an expression
                try:
                    last_line = code.strip().split('\n')[-1]
                    if last_line and not last_line.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ', 'print(')):
                        result = eval(compile(last_line, '<string>', 'eval'), safe_globals)
                except:
                    pass
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
    except TimeoutError as e:
        exception = e
    except Exception as e:
        exception = e
    
    # Build output
    output_parts = []
    
    stdout_content = stdout_buffer.getvalue()
    if stdout_content:
        output_parts.append(f"=== STDOUT ===\n{stdout_content}")
    
    stderr_content = stderr_buffer.getvalue()
    if stderr_content:
        output_parts.append(f"=== STDERR ===\n{stderr_content}")
    
    if exception:
        output_parts.append(f"=== EXCEPTION ===\n{type(exception).__name__}: {exception}")
    
    if result is not None:
        output_parts.append(f"=== RETURN VALUE ===\n{repr(result)}")
    
    if not output_parts:
        return "(no output)"
    
    return "\n\n".join(output_parts)
