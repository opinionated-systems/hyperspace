"""
Test runner tool: run Python tests to validate code changes.

Provides pytest and unittest execution capabilities to verify that
modifications don't break existing functionality.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "test_runner",
        "description": (
            "Run Python tests using pytest or unittest. "
            "Helps validate that code changes don't break existing functionality. "
            "Supports running specific test files, directories, or individual test cases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["pytest", "unittest", "discover"],
                    "description": "The test runner to use.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to test file or directory (absolute path).",
                },
                "test_name": {
                    "type": "string",
                    "description": "Optional specific test name to run (e.g., 'test_function' or 'TestClass.test_method').",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Whether to run tests with verbose output (default: true).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum time in seconds to wait for tests (default: 60).",
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


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Test operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_output(output: str, max_lines: int = 100) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def tool_function(
    command: str,
    path: str,
    test_name: str | None = None,
    verbose: bool = True,
    timeout: int = 60,
) -> str:
    """Execute tests using the specified runner."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if command == "pytest":
            return _run_pytest(p, test_name, verbose, timeout)
        elif command == "unittest":
            return _run_unittest(p, test_name, verbose, timeout)
        elif command == "discover":
            return _run_discover(p, verbose, timeout)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _run_pytest(path: Path, test_name: str | None, verbose: bool, timeout: int) -> str:
    """Run tests using pytest."""
    try:
        # First check if pytest is available
        check = subprocess.run(
            ["python", "-m", "pytest", "--version"],
            capture_output=True,
            timeout=5,
        )
        if check.returncode != 0:
            return "Error: pytest not found. Is it installed? Try: pip install pytest"
        
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        # Add test selection
        if test_name:
            cmd.append(f"{path}::{test_name}")
        else:
            cmd.append(str(path))
        
        # Add common options
        cmd.extend(["--tb=short", "--color=no"])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        
        # Summarize results
        summary = _summarize_results(result.returncode, output)
        
        return f"Test Results:\n{_truncate_output(output)}\n\n{summary}"
    except subprocess.TimeoutExpired:
        return f"Error: Tests timed out after {timeout}s"
    except FileNotFoundError:
        return "Error: pytest not found. Is it installed? Try: pip install pytest"
    except Exception as e:
        return f"Error running pytest: {e}"


def _run_unittest(path: Path, test_name: str | None, verbose: bool, timeout: int) -> str:
    """Run tests using unittest."""
    try:
        cmd = ["python", "-m", "unittest"]
        
        if verbose:
            cmd.append("-v")
        
        # Build test path
        if test_name:
            cmd.append(f"{path}.{test_name}")
        else:
            cmd.append(str(path))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        
        summary = _summarize_results(result.returncode, output)
        
        return f"Test Results:\n{_truncate_output(output)}\n\n{summary}"
    except subprocess.TimeoutExpired:
        return f"Error: Tests timed out after {timeout}s"
    except Exception as e:
        return f"Error running unittest: {e}"


def _run_discover(path: Path, verbose: bool, timeout: int) -> str:
    """Discover and run all tests in a directory."""
    try:
        cmd = ["python", "-m", "unittest", "discover", "-s", str(path)]
        
        if verbose:
            cmd.append("-v")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        
        summary = _summarize_results(result.returncode, output)
        
        return f"Test Results:\n{_truncate_output(output)}\n\n{summary}"
    except subprocess.TimeoutExpired:
        return f"Error: Tests timed out after {timeout}s"
    except Exception as e:
        return f"Error discovering tests: {e}"


def _summarize_results(returncode: int, output: str) -> str:
    """Generate a summary of test results."""
    if returncode == 0:
        return "✓ All tests passed"
    else:
        # Try to extract failure count
        failed = output.count("FAILED")
        errors = output.count("ERROR")
        
        if failed > 0 or errors > 0:
            return f"✗ Tests failed: {failed} failures, {errors} errors (exit code: {returncode})"
        else:
            return f"✗ Tests failed (exit code: {returncode})"
