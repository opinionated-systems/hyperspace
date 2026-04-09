"""
Test tool: run Python tests and check code validity.

Provides capabilities to run pytest, check Python syntax,
and validate that code changes don't break functionality.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "test",
        "description": (
            "Run Python tests and validate code. "
            "Commands: run_tests (run pytest), check_syntax (validate Python syntax), "
            "run_module (execute a Python module). "
            "Useful for verifying code changes work correctly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["run_tests", "check_syntax", "run_module"],
                    "description": "The test command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to test file, module, or directory (for run_tests).",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional arguments for pytest or module.",
                },
            },
            "required": ["command", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict test operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _truncate(content: str, max_len: int = 5000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def _check_path(path: str) -> tuple[Path | None, str]:
    """Validate and return the path, or return error message."""
    p = Path(path)
    
    if not p.is_absolute():
        return None, f"Error: {path} is not an absolute path."
    
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(_ALLOWED_ROOT):
            return None, f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    
    return p, ""


def tool_function(
    command: str,
    path: str,
    args: list[str] | None = None,
) -> str:
    """Execute a test command."""
    p, error = _check_path(path)
    if error:
        return error
    
    args = args or []
    
    if command == "run_tests":
        return _run_tests(p, args)
    elif command == "check_syntax":
        return _check_syntax(p)
    elif command == "run_module":
        return _run_module(p, args)
    else:
        return f"Error: unknown command {command}"


def _run_tests(p: Path, args: list[str]) -> str:
    """Run pytest on a file or directory."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", str(p), "-v", "--tb=short"] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_ALLOWED_ROOT or os.getcwd(),
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        
        if result.returncode == 0:
            return f"Tests passed!\n{_truncate(output, 8000)}"
        elif result.returncode == 5:
            return f"No tests found in {p}\n{_truncate(output, 4000)}"
        else:
            return f"Tests failed (exit code {result.returncode}):\n{_truncate(output, 8000)}"
            
    except subprocess.TimeoutExpired:
        return "Error: tests timed out after 120s"
    except Exception as e:
        return f"Error running tests: {e}"


def _check_syntax(p: Path) -> str:
    """Check Python syntax of a file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if not p.is_file():
        return f"Error: {p} is not a file."
    
    try:
        content = p.read_text()
    except Exception as e:
        return f"Error reading {p}: {e}"
    
    try:
        ast.parse(content)
        return f"Syntax OK: {p}"
    except SyntaxError as e:
        return f"Syntax error in {p} at line {e.lineno}, col {e.offset}: {e.msg}"
    except Exception as e:
        return f"Error parsing {p}: {e}"


def _run_module(p: Path, args: list[str]) -> str:
    """Run a Python module or file."""
    if not p.exists():
        return f"Error: {p} does not exist."
    
    if p.is_dir():
        # Try to run as a module
        module_name = p.name
        cmd = [sys.executable, "-m", module_name] + args
    else:
        # Run the file directly
        cmd = [sys.executable, str(p)] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_ALLOWED_ROOT or os.getcwd(),
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        
        if result.returncode == 0:
            return f"Module executed successfully:\n{_truncate(output, 8000)}"
        else:
            return f"Module exited with code {result.returncode}:\n{_truncate(output, 8000)}"
            
    except subprocess.TimeoutExpired:
        return "Error: module execution timed out after 60s"
    except Exception as e:
        return f"Error running module: {e}"
