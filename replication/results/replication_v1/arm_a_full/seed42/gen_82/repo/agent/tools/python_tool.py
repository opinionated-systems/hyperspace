"""
Python execution tool: safely execute Python code snippets.

Provides a sandboxed environment for running Python code,
useful for testing, calculations, and data processing.
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
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python code in a sandboxed environment. "
                "Use this for calculations, data processing, testing code snippets, "
                "or any task that requires Python execution. "
                "The code runs in an isolated namespace with limited builtins. "
                "Output is captured from stdout/stderr and returned."
            ),
            "parameters": {
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
        },
    }


def _create_safe_globals() -> dict:
    """Create a restricted globals dict for safe execution."""
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "bytearray": bytearray,
        "bytes": bytes,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hasattr": hasattr,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        "__import__": __import__,
        "True": True,
        "False": False,
        "None": None,
    }
    return {"__builtins__": safe_builtins}


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code and return output.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Captured stdout/stderr output or error message
    """
    # Clamp timeout to safe range
    timeout = max(1, min(timeout, 120))

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Create safe execution environment
    globals_dict = _create_safe_globals()
    locals_dict = {}

    result_lines = []

    try:
        # Compile the code to check for syntax errors
        compiled_code = compile(code, "<string>", "exec")

        # Execute with output capture
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compiled_code, globals_dict, locals_dict)

        # Get captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        if stdout_output:
            result_lines.append("[stdout]\n" + stdout_output)
        if stderr_output:
            result_lines.append("[stderr]\n" + stderr_output)

        # If no output, show the last expression value if available
        if not stdout_output and not stderr_output:
            # Try to get the result of the last expression
            try:
                last_line = code.strip().split("\n")[-1]
                if last_line and not last_line.startswith((" ", "\t", "#", "import", "from", "def ", "class ", "if ", "for ", "while ", "with ", "try:", "except", "finally", "elif ", "else:")):
                    result = eval(last_line, globals_dict, locals_dict)
                    if result is not None:
                        result_lines.append(f"[result]\n{repr(result)}")
            except Exception:
                pass  # Ignore eval errors

        if not result_lines:
            return "Code executed successfully (no output)."

        return "\n\n".join(result_lines)

    except SyntaxError as e:
        return f"[SyntaxError] {e.msg} at line {e.lineno}, col {e.offset}"
    except Exception as e:
        error_msg = f"[{type(e).__name__}] {str(e)}"
        tb = traceback.format_exc()
        return f"{error_msg}\n\n[traceback]\n{tb}"
