"""
Python evaluation tool: runs Python code in a subprocess.

The tool accepts a string of Python code and returns the stdout output.
It uses a temporary file to run the code with `python -c`.

The tool is intentionally simple and does not provide a sandbox.
"""

from __future__ import annotations

import subprocess
import uuid
import os


def tool_info() -> dict:
    return {
        "name": "python",
        "description": (
            "Run arbitrary Python code in a subprocess. "
            "The code is executed with `python -c` and stdout is returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                }
            },
            "required": ["code"],
        },
    }


_TIMEOUT = 30.0


def tool_function(code: str) -> str:
    """Execute the provided Python code and return stdout.

    The code is executed in a temporary file to avoid issues with
    quoting and to provide a clean environment.
    """
    tmp_file = f"/tmp/python_tool_{uuid.uuid4().hex}.py"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            ["python", tmp_file],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error executing Python code: {e}"
    finally:
        try:
            os.remove(tmp_file)
        except Exception:
            pass
