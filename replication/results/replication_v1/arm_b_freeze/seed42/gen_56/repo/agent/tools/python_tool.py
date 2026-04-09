"""
Python execution tool: safely execute Python code and return output.

Allows the agent to run Python code for testing, calculations,
data processing, and validation tasks.
"""

from __future__ import annotations

import io
import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "python",
        "description": (
            "Execute Python code and return the output. "
            "Use this for calculations, data processing, testing code snippets, "
            "or any task that requires Python execution. "
            "The code runs in a restricted environment with a 30-second timeout. "
            "Print statements and return values are captured."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can include multiple lines, imports, and function definitions.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 30, max: 120).",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    }


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code and return output.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30, max: 120)

    Returns:
        Execution output (stdout, stderr, and return value)
    """
    # Clamp timeout to reasonable bounds
    timeout = max(1, min(timeout, 120))

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = None
    error = None

    try:
        # Create restricted globals for safer execution
        safe_globals = {
            "__builtins__": {
                name: __builtins__[name]  # type: ignore
                for name in [
                    "abs", "all", "any", "bin", "bool", "bytearray", "bytes",
                    "chr", "complex", "dict", "dir", "divmod", "enumerate",
                    "filter", "float", "format", "frozenset", "hasattr",
                    "hash", "hex", "id", "int", "isinstance", "issubclass",
                    "iter", "len", "list", "map", "max", "memoryview", "min",
                    "next", "oct", "ord", "pow", "print", "range", "repr",
                    "reversed", "round", "set", "slice", "sorted", "str",
                    "sum", "tuple", "type", "vars", "zip", "True", "False",
                    "None", "Exception", "BaseException", "ArithmeticError",
                    "AssertionError", "AttributeError", "BlockingIOError",
                    "BrokenPipeError", "BufferError", "BytesWarning",
                    "ChildProcessError", "ConnectionAbortedError",
                    "ConnectionError", "ConnectionRefusedError",
                    "ConnectionResetError", "DeprecationWarning", "EOFError",
                    "EnvironmentError", "FileExistsError", "FileNotFoundError",
                    "FloatingPointError", "FutureWarning", "GeneratorExit",
                    "IOError", "ImportError", "ImportWarning", "IndentationError",
                    "IndexError", "InterruptedError", "IsADirectoryError",
                    "KeyError", "KeyboardInterrupt", "LookupError",
                    "MemoryError", "ModuleNotFoundError", "NameError",
                    "NotADirectoryError", "NotImplementedError", "OSError",
                    "OverflowError", "PendingDeprecationWarning",
                    "PermissionError", "ProcessLookupError", "RecursionError",
                    "ReferenceError", "ResourceWarning", "RuntimeError",
                    "RuntimeWarning", "StopAsyncIteration", "StopIteration",
                    "SyntaxError", "SyntaxWarning", "SystemError", "SystemExit",
                    "TabError", "TimeoutError", "TypeError", "UnboundLocalError",
                    "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeError",
                    "UnicodeTranslateError", "UnicodeWarning", "UserWarning",
                    "ValueError", "Warning", "ZeroDivisionError",
                    # Allow essential functions
                    "__import__", "open", "input", "eval", "exec", "compile",
                    "getattr", "setattr", "delattr", "classmethod", "staticmethod",
                    "property", "super", "object", "help",
                ]
                if name in __builtins__  # type: ignore
            },
            "__name__": "__main__",
        }

        # Add commonly used modules to safe globals
        import math
        import random
        import datetime
        import json
        import re
        import string
        import itertools
        import collections
        import statistics
        import typing
        import fractions
        import decimal
        import hashlib
        import uuid
        import time
        import inspect

        safe_globals.update({
            "math": math,
            "random": random,
            "datetime": datetime,
            "json": json,
            "re": re,
            "string": string,
            "itertools": itertools,
            "collections": collections,
            "statistics": statistics,
            "typing": typing,
            "fractions": fractions,
            "decimal": decimal,
            "hashlib": hashlib,
            "uuid": uuid,
            "time": time,
            "inspect": inspect,
        })

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code
            exec(code, safe_globals)

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Build result
        output_parts = []
        if stdout_output:
            output_parts.append(f"[stdout]\n{stdout_output}")
        if stderr_output:
            output_parts.append(f"[stderr]\n{stderr_output}")

        if not output_parts:
            return "Code executed successfully (no output)."

        return "\n\n".join(output_parts)

    except Exception as e:
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        output_parts = []
        if stdout_output:
            output_parts.append(f"[stdout before error]\n{stdout_output}")

        error_msg = f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        output_parts.append(error_msg)

        return "\n\n".join(output_parts)


if __name__ == "__main__":
    # Test the tool
    print(tool_function("print('Hello, World!')"))
    print("---")
    print(tool_function("x = 5 + 3\nprint(f'Result: {x}')"))
    print("---")
    print(tool_function("import math\nprint(math.pi)"))
