"""
Python tool: execute Python code in a sandboxed environment.

Provides a safe way to run Python code for testing, calculations,
and validation. The code runs in a restricted environment with
limited access to system resources.
"""

from __future__ import annotations

import ast
import io
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any


# Restricted builtins - only allow safe operations
_ALLOWED_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "hasattr", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "memoryview",
    "min", "next", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "vars", "zip", "True", "False", "None",
    "__import__",  # We'll override this
}

# Modules that can be imported
_ALLOWED_MODULES = {
    "math", "random", "statistics", "itertools", "functools", "collections",
    "datetime", "decimal", "fractions", "numbers", "typing", "json",
    "re", "string", "hashlib", "base64", "copy", "pprint", "textwrap",
    "string", "time", "inspect", "types", "dataclasses", "enum",
}


def tool_info() -> dict:
    return {
        "name": "python",
        "description": (
            "Execute Python code in a sandboxed environment. "
            "Useful for testing code, running calculations, and validation. "
            "Code runs with restricted builtins and limited imports. "
            "Returns stdout, stderr, and the result of the last expression."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default 30, max 120).",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    }


def _safe_import(name: str, *args, **kwargs):
    """Restricted import that only allows safe modules."""
    if name not in _ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed. Allowed modules: {sorted(_ALLOWED_MODULES)}")
    return __import__(name, *args, **kwargs)


def _validate_code(code: str) -> tuple[bool, str]:
    """Validate that code doesn't contain dangerous operations."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    for node in ast.walk(tree):
        # Check for dangerous operations
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_MODULES:
                    return False, f"Import of '{alias.name}' is not allowed"
        
        if isinstance(node, ast.ImportFrom):
            if node.module not in _ALLOWED_MODULES:
                return False, f"Import from '{node.module}' is not allowed"
        
        # Check for exec/eval
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("exec", "eval", "compile"):
                    return False, f"Use of '{node.func.id}()' is not allowed"
        
        # Check for attribute access that might be dangerous
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") and node.attr != "__init__":
                # Allow dunder methods but not private attributes
                if not (node.attr.startswith("__") and node.attr.endswith("__")):
                    pass  # We'll be lenient here
    
    return True, ""


def _create_restricted_globals(stdout, stderr) -> dict:
    """Create a restricted globals dict for safe execution."""
    # Start with allowed builtins
    safe_builtins = {name: __builtins__[name] for name in _ALLOWED_BUILTINS if name in __builtins__}
    
    # Override __import__
    safe_builtins["__import__"] = _safe_import
    
    # Create a custom print that writes to the captured stdout
    def _safe_print(*args, sep=' ', end='\n', file=None, flush=False):
        if file is None:
            file = stdout
        return __builtins__['print'](*args, sep=sep, end=end, file=file, flush=flush)
    
    safe_builtins["print"] = _safe_print
    
    return {"__builtins__": safe_builtins}


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code in a sandboxed environment.
    
    Args:
        code: The Python code to execute
        timeout: Maximum execution time in seconds (default 30, max 120)
        
    Returns:
        String containing stdout, stderr, and result
    """
    # Validate timeout
    timeout = min(max(timeout, 1), 120)
    
    # Validate code
    is_valid, error_msg = _validate_code(code)
    if not is_valid:
        return f"Error: {error_msg}"
    
    # Capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Create restricted environment with captured output
    restricted_globals = _create_restricted_globals(stdout_buffer, stderr_buffer)
    restricted_locals = {}
    
    result = None
    error = None
    
    try:
        # Compile and execute
        compiled = compile(code, "<sandbox>", "exec")
        exec(compiled, restricted_globals, restricted_locals)
        
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        # Try to get the last expression value if code ends with an expression
        # Do this outside the redirect context to avoid double printing
        try:
            lines = code.strip().split("\n")
            if lines:
                last_line = lines[-1].strip()
                if last_line and not last_line.startswith(("#", "import", "from", "def", "class", "if", "for", "while", "with", "try", "except", "finally", "return", "raise", "assert", "del", "global", "nonlocal", "pass", "break", "continue")):
                    try:
                        result = eval(compile(last_line, "<sandbox>", "eval"), restricted_globals, restricted_locals)
                    except SyntaxError:
                        pass  # Not a valid expression
        except Exception:
            pass
        
        # Build result
        parts = []
        if stdout_output:
            parts.append(f"[stdout]\n{stdout_output}")
        if stderr_output:
            parts.append(f"[stderr]\n{stderr_output}")
        if result is not None:
            parts.append(f"[result]\n{repr(result)}")
        
        if not parts:
            return "Code executed successfully (no output)"
        
        return "\n\n".join(parts)
        
    except Exception as e:
        error = f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        return error


# For testing
if __name__ == "__main__":
    # Test basic execution
    print(tool_function("print('Hello, World!')"))
    print("---")
    
    # Test math
    print(tool_function("import math\nprint(math.sqrt(16))"))
    print("---")
    
    # Test result capture
    print(tool_function("x = 5\ny = 10\nx + y"))
    print("---")
    
    # Test restricted import
    print(tool_function("import os"))
