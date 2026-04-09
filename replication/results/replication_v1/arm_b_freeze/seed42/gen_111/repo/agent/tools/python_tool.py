"""
Python execution tool: safely execute Python code in a sandboxed environment.

Provides a way for the agent to test code, run calculations, and validate
changes before applying them to the codebase.
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
    """Return tool specification for OpenAI function calling."""
    return {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code in a sandboxed environment. "
                "Use this to test code, run calculations, or validate changes. "
                "The code runs in an isolated namespace with limited builtins. "
                "Returns stdout, stderr, and any return value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Can include multiple lines and imports.",
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


def _create_safe_globals() -> dict[str, Any]:
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
    }
    
    # Allow common safe modules
    safe_modules = {}
    allowed_modules = [
        "math", "random", "statistics", "itertools", "collections",
        "functools", "datetime", "json", "re", "string", "typing",
        "decimal", "fractions", "hashlib", "uuid", "copy", "pprint",
        "textwrap", "inspect", "types", "dataclasses", "enum",
    ]
    
    for mod_name in allowed_modules:
        try:
            safe_modules[mod_name] = __import__(mod_name)
        except ImportError:
            pass
    
    return {"__builtins__": safe_builtins, **safe_modules}


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code safely and return the result.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
        
    Returns:
        String containing stdout, stderr, return value, and any errors
    """
    import signal
    
    # Clamp timeout to reasonable bounds
    timeout = max(1, min(timeout, 120))
    
    # Create isolated namespace
    globals_dict = _create_safe_globals()
    locals_dict = {}
    
    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    result_parts = []
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {timeout} seconds")
    
    # Set up timeout (Unix only)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    except (AttributeError, ValueError):
        # Windows or signal not available
        pass
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Compile and execute
            compiled = compile(code, "<string>", "exec")
            exec(compiled, globals_dict, locals_dict)
        
        # Get captured output
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        # Build result
        if stdout:
            result_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            result_parts.append(f"STDERR:\n{stderr}")
        
        # Check for a return value (if last statement was an expression)
        if locals_dict:
            result_parts.append(f"Local variables: {list(locals_dict.keys())}")
        
        result = "\n\n".join(result_parts) if result_parts else "Code executed successfully (no output)"
        
    except TimeoutError as e:
        result = f"ERROR: Timeout - {e}"
    except SyntaxError as e:
        result = f"ERROR: Syntax error at line {e.lineno}: {e.msg}\n{e.text}"
    except Exception as e:
        result = f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        # Reset alarm
        try:
            signal.alarm(0)
            if old_handler:
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass
    
    logger.info("Python tool executed code (%d chars), result: %s", len(code), result[:200])
    return result
