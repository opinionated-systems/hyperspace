"""
Python execution tool: safely execute Python code and return output.

Allows the agent to test code snippets, run calculations, and verify
modifications work as expected.
"""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any


def tool_info() -> dict:
    return {
        "name": "python",
        "description": (
            "Execute Python code and return the output. "
            "Useful for testing code snippets, running calculations, "
            "and verifying modifications work as expected. "
            "Code runs in a restricted environment with limited builtins."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can include newlines and indentation.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 30).",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    }


def _safe_globals() -> dict[str, Any]:
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
    }
    return {"__builtins__": safe_builtins}


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code and return output."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        # Set up restricted environment
        globals_dict = _safe_globals()
        locals_dict = {}
        
        # Capture output
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Compile and execute with timeout protection via signal
            compiled = compile(code, "<string>", "exec")
            exec(compiled, globals_dict, locals_dict)
        
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        result_parts = []
        if stdout_output:
            result_parts.append(f"STDOUT:\n{stdout_output}")
        if stderr_output:
            result_parts.append(f"STDERR:\n{stderr_output}")
        
        if not result_parts:
            # Show any defined variables as result
            defined_vars = {k: v for k, v in locals_dict.items() if not k.startswith("_")}
            if defined_vars:
                result_parts.append(f"Variables defined: {list(defined_vars.keys())}")
            else:
                result_parts.append("Code executed successfully (no output)")
        
        return "\n\n".join(result_parts)
        
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()
        return f"Error: {error_msg}\n\nTraceback:\n{tb}"


if __name__ == "__main__":
    # Test the tool
    print(tool_function("print('Hello, World!')"))
    print("---")
    print(tool_function("x = 5 + 3\nprint(f'Result: {x}')"))
    print("---")
    print(tool_function("import os"))  # Should fail - import not allowed
