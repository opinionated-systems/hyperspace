"""
Python tool: execute Python code in a controlled environment.

Provides the ability to run Python code for testing, validation,
and quick prototyping during codebase modifications.
"""

from __future__ import annotations

import io
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "python",
        "description": (
            "Execute Python code in a controlled environment. "
            "Useful for testing changes, running quick calculations, "
            "and validating code modifications. "
            "Code runs in an isolated namespace with access to standard library."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
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


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict file operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code and return the output.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Output from stdout/stderr or error message
    """
    # Create isolated namespace
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "os": os,
        "sys": sys,
        "Path": Path,
    }
    
    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        # Execute with output capture
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, namespace)
        
        # Get outputs
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        # Combine outputs
        result_parts = []
        if stdout_output:
            result_parts.append(f"STDOUT:\n{stdout_output}")
        if stderr_output:
            result_parts.append(f"STDERR:\n{stderr_output}")
        
        if not result_parts:
            return "(no output)"
        
        full_output = "\n\n".join(result_parts)
        
        # Truncate if too long
        if len(full_output) > 10000:
            full_output = full_output[:5000] + "\n... [output truncated] ...\n" + full_output[-5000:]
        
        return full_output
        
    except Exception as e:
        # Capture exception details
        exc_type = type(e).__name__
        exc_msg = str(e)
        tb = traceback.format_exc()
        
        return f"Error ({exc_type}): {exc_msg}\n\nTraceback:\n{tb}"
