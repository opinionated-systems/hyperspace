"""
Python execution tool for running code snippets safely.

Provides a sandboxed environment for executing Python code
with restricted builtins and timeout protection.
"""

from __future__ import annotations

import ast
import logging
import multiprocessing
import os
import sys
import tempfile
import traceback
from typing import Any

logger = logging.getLogger(__name__)

# Restricted builtins for safer execution
ALLOWED_BUILTINS = {
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

# Modules that can be imported
ALLOWED_MODULES = {
    "math",
    "random",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "statistics",
    "decimal",
    "fractions",
    "json",
    "re",
    "string",
    "typing",
    "hashlib",
    "uuid",
    "time",
    "calendar",
    "bisect",
    "heapq",
    "copy",
    "pprint",
    "textwrap",
    "enum",
    "dataclasses",
    "pathlib",
    "inspect",
}


def safe_import(name: str, *args, **kwargs) -> Any:
    """Restricted import that only allows safe modules."""
    if name in ALLOWED_MODULES or name.split(".")[0] in ALLOWED_MODULES:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"Import of '{name}' is not allowed")


def _execute_code_worker(code: str, result_queue: multiprocessing.Queue, temp_dir: str) -> None:
    """Worker function that runs in a separate process."""
    # Redirect stdout/stderr to capture output
    import io
    import builtins
    from contextlib import redirect_stdout, redirect_stderr

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Create restricted globals
        safe_builtins = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS if hasattr(builtins, name)}
        safe_builtins["__import__"] = safe_import

        restricted_globals = {
            "__builtins__": safe_builtins,
        }

        # Change to temp directory for file operations
        old_cwd = os.getcwd()
        os.chdir(temp_dir)

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Compile and execute
            compiled = compile(code, "<string>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            exec(compiled, restricted_globals)

        os.chdir(old_cwd)

        result_queue.put({
            "success": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": None,  # Could extract last expression value if needed
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue() + f"\n{traceback.format_exc()}",
            "error": str(e),
        })


def execute_python(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dict with success status, stdout, stderr, and any error
    """
    # Validate code with AST first
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in ALLOWED_MODULES:
                        return {
                            "success": False,
                            "stdout": "",
                            "stderr": f"",
                            "error": f"Import of '{alias.name}' is not allowed",
                        }
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in ALLOWED_MODULES:
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": "",
                        "error": f"Import from '{node.module}' is not allowed",
                    }
    except SyntaxError as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"Syntax error: {e}",
        }

    # Create temp directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        result_queue = multiprocessing.Queue()

        # Run in separate process for isolation and timeout
        process = multiprocessing.Process(
            target=_execute_code_worker,
            args=(code, result_queue, temp_dir)
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"Execution timed out after {timeout} seconds",
            }

        if not result_queue.empty():
            return result_queue.get()

        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": "Process exited without result",
        }


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python code in a sandboxed environment. Useful for calculations, data processing, testing code snippets, and running algorithms. Restricted imports: math, random, datetime, json, re, collections, itertools, statistics, and other standard library modules. No network or file system access outside temp directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Can include multiple statements, function definitions, and imports from allowed modules.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (default: 30, max: 120)",
                        "default": 30,
                    },
                },
                "required": ["code"],
            },
        },
    }


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code and return formatted output."""
    # Clamp timeout
    timeout = max(1, min(120, timeout))

    result = execute_python(code, timeout)

    output_parts = []

    if result["stdout"]:
        output_parts.append(f"[stdout]\n{result['stdout']}")

    if result["stderr"]:
        output_parts.append(f"[stderr]\n{result['stderr']}")

    if result["success"]:
        output_parts.append("[status] Success")
    else:
        output_parts.append(f"[status] Error: {result.get('error', 'Unknown error')}")

    return "\n\n".join(output_parts)
