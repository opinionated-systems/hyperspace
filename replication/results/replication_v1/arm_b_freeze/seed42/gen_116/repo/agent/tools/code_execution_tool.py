"""
Code execution tool: safely execute Python code in a sandboxed environment.

Provides a secure way to run Python code with resource limits and timeout protection.
Useful for testing code snippets, running calculations, or validating implementations.
"""

from __future__ import annotations

import ast
import resource
import signal
import sys
import traceback
from io import StringIO
from typing import Any


def _set_resource_limits():
    """Set resource limits for the sandboxed execution."""
    # Limit memory to 256MB
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    # Limit CPU time to 5 seconds
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))


def _timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Code execution timed out (max 5 seconds)")


def _is_safe_code(code: str) -> tuple[bool, str]:
    """Check if code is safe to execute by analyzing the AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # List of forbidden node types
    forbidden_nodes = (
        ast.Import,  # No imports allowed
        ast.ImportFrom,  # No imports allowed
        ast.Call,  # Check calls separately
    )
    
    for node in ast.walk(tree):
        # Check for imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, f"Import statements are not allowed: {ast.dump(node)}"
        
        # Check for dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                dangerous_funcs = ['eval', 'exec', 'compile', '__import__', 'open', 'input']
                if node.func.id in dangerous_funcs:
                    return False, f"Dangerous function call not allowed: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                # Check for methods like os.system, subprocess.call, etc.
                dangerous_attrs = ['system', 'popen', 'call', 'run', 'check_output']
                if node.func.attr in dangerous_attrs:
                    return False, f"Dangerous method call not allowed: {node.func.attr}"
    
    return True, ""


def _execute_code(code: str) -> dict[str, Any]:
    """Execute Python code safely with resource limits."""
    # First, check if code is safe
    is_safe, error_msg = _is_safe_code(code)
    if not is_safe:
        return {"error": error_msg}
    
    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)  # 5 second timeout
    
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    result = {
        "output": "",
        "error": None,
        "stdout": "",
        "stderr": "",
    }
    
    try:
        # Set resource limits in the child process
        _set_resource_limits()
        
        # Execute the code
        exec(code, {"__builtins__": __builtins__}, {})
        
        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue()
        
        if result["stderr"]:
            result["error"] = result["stderr"]
        else:
            result["output"] = result["stdout"] if result["stdout"] else "Code executed successfully (no output)"
            
    except TimeoutError as e:
        result["error"] = str(e)
    except MemoryError:
        result["error"] = "Memory limit exceeded (max 256MB)"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Restore signal handler and streams
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "code_execution",
        "description": "Execute Python code safely in a sandboxed environment with resource limits (256MB memory, 5 second timeout). Useful for testing code snippets, running calculations, or validating implementations. No imports or dangerous functions allowed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Must be pure Python without imports. Dangerous functions (eval, exec, open, etc.) are blocked.",
                },
            },
            "required": ["code"],
        },
    }


def tool_function(code: str) -> str:
    """Execute Python code safely."""
    result = _execute_code(code)
    
    if result.get("error"):
        return f"Error: {result['error']}"
    
    output = result.get("output", "")
    # Truncate very long outputs
    if len(output) > 10000:
        output = output[:5000] + "\n... [output truncated] ...\n" + output[-5000:]
    
    return output
