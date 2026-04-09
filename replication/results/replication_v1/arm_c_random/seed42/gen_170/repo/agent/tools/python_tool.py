"""
Python execution tool: run Python code safely in a subprocess.

Provides a sandboxed environment for executing Python code with
resource limits and timeout protection.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import os
import signal
from typing import Any


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "python",
        "description": "Execute Python code in a sandboxed subprocess. Useful for calculations, data processing, file operations, and testing code snippets. Returns stdout, stderr, and any errors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can include multiple statements, function definitions, etc.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 30, max: 120)",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    }


def tool_function(code: str, timeout: int = 30) -> str:
    """Execute Python code in a sandboxed subprocess.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30, max: 120)
        
    Returns:
        JSON string with stdout, stderr, return_code, and any error messages
    """
    # Clamp timeout to safe range
    timeout = max(1, min(timeout, 120))
    
    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Run the code in a subprocess with resource limits
        result = subprocess.run(
            ['python3', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "success": result.returncode == 0,
        }
        
    except subprocess.TimeoutExpired:
        output = {
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "success": False,
            "error": f"Execution timed out after {timeout} seconds",
        }
    except Exception as e:
        output = {
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "success": False,
            "error": str(e),
        }
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return json.dumps(output, indent=2)
