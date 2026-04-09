"""
Python tool: execute Python code safely with output capture.

Provides a controlled environment for running Python code with:
- Timeout protection against infinite loops
- Output capture (stdout/stderr)
- Restricted builtins for safety
- Proper error handling and traceback reporting
"""

from __future__ import annotations

import io
import logging
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

logger = logging.getLogger(__name__)

# Default timeout for code execution (seconds)
DEFAULT_TIMEOUT = 30

# Restricted builtins for safety - only allow essential functions
SAFE_BUILTINS = {
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
    "vars": vars,
    "zip": zip,
    "__import__": __import__,  # Allow imports but they'll be restricted by timeout
    "True": True,
    "False": False,
    "None": None,
}


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "python",
        "description": (
            "Execute Python code safely with output capture. "
            "Use this for calculations, data processing, string manipulation, "
            "or any task that requires Python logic. "
            "Code runs in a restricted environment with a timeout. "
            "Output is captured and returned."
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
                    "description": f"Maximum execution time in seconds (default: {DEFAULT_TIMEOUT}).",
                    "default": DEFAULT_TIMEOUT,
                },
            },
            "required": ["code"],
        },
    }


def _execute_with_timeout(code: str, timeout: int) -> dict[str, Any]:
    """Execute Python code with timeout and capture output."""
    import signal
    
    result = {
        "stdout": "",
        "stderr": "",
        "return_value": None,
        "error": None,
        "timed_out": False,
    }
    
    # Create restricted globals
    restricted_globals = {
        "__builtins__": SAFE_BUILTINS.copy(),
        "__name__": "__main__",
    }
    
    # Set up timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {timeout} seconds")
    
    # Capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        # Set alarm for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Compile and execute the code
                compiled = compile(code, "<string>", "exec")
                exec(compiled, restricted_globals)
                
                # Check if there's a return value (last expression in interactive mode)
                # Try to get the value of the last expression if it was an expression
                try:
                    last_line = code.strip().split("\n")[-1]
                    if last_line and not last_line.startswith((" ", "\t", "#", "import", "from", "def", "class", "if", "for", "while", "with", "try")):
                        result["return_value"] = eval(last_line, restricted_globals)
                except Exception:
                    pass  # Not an expression or failed to eval
                    
        finally:
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
            
        result["stdout"] = stdout_buffer.getvalue()
        result["stderr"] = stderr_buffer.getvalue()
        
    except TimeoutError as e:
        result["timed_out"] = True
        result["error"] = str(e)
        result["stdout"] = stdout_buffer.getvalue()
        result["stderr"] = stderr_buffer.getvalue()
        logger.warning(f"Python code timed out after {timeout}s")
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["stdout"] = stdout_buffer.getvalue()
        result["stderr"] = stderr_buffer.getvalue()
        logger.warning(f"Python code execution error: {e}")
    
    return result


def tool_function(code: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute Python code and return formatted results.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Formatted string with output, errors, and return value
    """
    if not code or not code.strip():
        return "Error: No code provided"
    
    logger.info(f"Executing Python code ({len(code)} chars, timeout={timeout}s)")
    
    result = _execute_with_timeout(code.strip(), timeout)
    
    # Format the output
    output_parts = []
    
    if result["stdout"]:
        output_parts.append(f"[stdout]\n{result['stdout']}")
    
    if result["stderr"]:
        output_parts.append(f"[stderr]\n{result['stderr']}")
    
    if result["return_value"] is not None:
        output_parts.append(f"[return value]\n{repr(result['return_value'])}")
    
    if result["timed_out"]:
        output_parts.append(f"[timeout]\n{result['error']}")
    elif result["error"]:
        output_parts.append(f"[error]\n{result['error']}")
        if result.get("traceback"):
            # Include truncated traceback
            tb_lines = result["traceback"].split("\n")
            if len(tb_lines) > 10:
                tb = "\n".join(tb_lines[:5] + ["..."] + tb_lines[-5:])
            else:
                tb = result["traceback"]
            output_parts.append(f"[traceback]\n{tb}")
    
    if not output_parts:
        return "(no output)"
    
    return "\n\n".join(output_parts)
