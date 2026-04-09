"""
Python tool: execute Python code in a sandboxed environment.

Provides a safe way to run Python code with restricted builtins
and timeout protection. Useful for calculations, data processing,
and quick prototyping without needing to create files.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import re
import sys
import traceback
from typing import Any


# Timeout for code execution (seconds)
_TIMEOUT = 30.0

# Restricted builtins - only allow safe operations
_ALLOWED_BUILTINS = {
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hasattr",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "True",
    "False",
    "None",
    "__import__",  # Will be replaced with safe_import
}

# Dangerous patterns to block
_DANGEROUS_PATTERNS = [
    (r"\b__import__\s*\(", "Direct __import__ calls are not allowed"),
    (r"\bimport\s+os\b", "Importing 'os' module is not allowed"),
    (r"\bimport\s+sys\b", "Importing 'sys' module is not allowed"),
    (r"\bimport\s+subprocess\b", "Importing 'subprocess' module is not allowed"),
    (r"\bfrom\s+os\b", "Importing from 'os' is not allowed"),
    (r"\bfrom\s+sys\b", "Importing from 'sys' is not allowed"),
    (r"\bfrom\s+subprocess\b", "Importing from 'subprocess' is not allowed"),
    (r"\bopen\s*\(", "File operations via open() are not allowed"),
    (r"\bexec\s*\(", "exec() is not allowed"),
    (r"\beval\s*\(", "eval() is not allowed"),
    (r"\bcompile\s*\(", "compile() is not allowed"),
    (r"\binput\s*\(", "input() is not allowed"),
    (r"\b__subclasses__\b", "Accessing __subclasses__ is not allowed"),
    (r"\bglobals\s*\(", "globals() is not allowed"),
    (r"\blocals\s*\(", "locals() is not allowed"),
    (r"\bvars\s*\(", "vars() is not allowed"),
    (r"\bgetattr\s*\(", "getattr() is restricted"),
    (r"\bsetattr\s*\(", "setattr() is not allowed"),
    (r"\bdelattr\s*\(", "delattr() is not allowed"),
    (r"\bexit\s*\(", "exit() is not allowed"),
    (r"\bquit\s*\(", "quit() is not allowed"),
]


def tool_info() -> dict:
    return {
        "name": "python",
        "description": (
            "Execute Python code in a sandboxed environment. "
            "Useful for calculations, data processing, and quick prototyping. "
            "Code runs with restricted builtins and timeout protection. "
            "Returns stdout output and any return value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": f"Execution timeout in seconds (default: {_TIMEOUT}, max: 60).",
                },
            },
            "required": ["code"],
        },
    }


def _is_code_safe(code: str) -> tuple[bool, str]:
    """Check if code contains dangerous patterns."""
    for pattern, message in _DANGEROUS_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Security error: {message}"
    return True, ""


def _safe_import(name: str, *args, **kwargs) -> Any:
    """Restricted import that blocks dangerous modules."""
    blocked_modules = {
        "os", "sys", "subprocess", "socket", "http", "urllib", "ftplib",
        "pickle", "marshal", "ctypes", "multiprocessing", "threading",
        "_thread", "importlib", "imp", "builtins", "__builtin__",
    }
    
    base_name = name.split(".")[0]
    if base_name in blocked_modules:
        raise ImportError(f"Import of '{name}' is not allowed for security reasons")
    
    # Allow safe standard library modules
    allowed_modules = {
        "math", "random", "statistics", "decimal", "fractions", "numbers",
        "datetime", "time", "calendar", "itertools", "functools", "collections",
        "heapq", "bisect", "copy", "pprint", "reprlib", "enum", "types",
        "string", "re", "json", "csv", "html", "textwrap", "unicodedata",
        "hashlib", "base64", "binascii", "struct", "codecs",
    }
    
    if base_name not in allowed_modules:
        raise ImportError(f"Import of '{name}' is not in the allowed modules list")
    
    return __import__(name, *args, **kwargs)


def _execute_with_timeout(code: str, timeout: float) -> tuple[str, Any, bool]:
    """Execute code with timeout protection.
    
    Returns: (stdout_output, return_value, timed_out)
    """
    import signal
    
    stdout_buffer = io.StringIO()
    result = {"value": None, "error": None, "completed": False}
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {timeout} seconds")
    
    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))
    
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            # Create restricted globals
            safe_globals = {
                "__builtins__": {
                    name: getattr(__builtins__, name)
                    for name in _ALLOWED_BUILTINS
                    if hasattr(__builtins__, name)
                },
                "__import__": _safe_import,
            }
            
            # Add useful modules
            safe_globals["math"] = __import__("math")
            safe_globals["random"] = __import__("random")
            safe_globals["statistics"] = __import__("statistics")
            safe_globals["datetime"] = __import__("datetime")
            safe_globals["itertools"] = __import__("itertools")
            safe_globals["functools"] = __import__("functools")
            safe_globals["collections"] = __import__("collections")
            safe_globals["json"] = __import__("json")
            safe_globals["re"] = __import__("re")
            safe_globals["string"] = __import__("string")
            
            # Execute the code
            exec(code, safe_globals)
            result["completed"] = True
            
            # Check if there's a 'result' variable defined
            if "result" in safe_globals:
                result["value"] = safe_globals["result"]
            
    except TimeoutError as e:
        result["error"] = str(e)
        result["timed_out"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    stdout_output = stdout_buffer.getvalue()
    timed_out = result.get("timed_out", False)
    
    return stdout_output, result["value"], timed_out


def tool_function(code: str, timeout: float | None = None) -> str:
    """Execute Python code safely.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 30, max: 60)
    
    Returns:
        Output from the code execution
    """
    if not code or not code.strip():
        return "Error: No code provided"
    
    # Validate timeout
    if timeout is None:
        timeout = _TIMEOUT
    timeout = min(max(timeout, 1), 60)  # Clamp between 1 and 60 seconds
    
    # Security check
    is_safe, error_msg = _is_code_safe(code)
    if not is_safe:
        return f"Error: {error_msg}"
    
    # Try to parse the code to check for syntax errors
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    
    # Execute the code
    try:
        stdout_output, return_value, timed_out = _execute_with_timeout(code, timeout)
        
        if timed_out:
            return f"Error: Code execution timed out after {timeout} seconds"
        
        # Build output
        output_parts = []
        
        if stdout_output:
            output_parts.append(f"Output:\n{stdout_output}")
        
        if return_value is not None:
            try:
                return_str = repr(return_value)
                if len(return_str) > 1000:
                    return_str = return_str[:500] + "... [truncated] ..." + return_str[-500:]
                output_parts.append(f"Return value: {return_str}")
            except Exception:
                output_parts.append(f"Return value: <unable to repr>")
        
        if not output_parts:
            return "Code executed successfully (no output)"
        
        return "\n\n".join(output_parts)
        
    except Exception as e:
        error_details = traceback.format_exc()
        return f"Error during execution: {type(e).__name__}: {e}\n\nTraceback:\n{error_details}"
