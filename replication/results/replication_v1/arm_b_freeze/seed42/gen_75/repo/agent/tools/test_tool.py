"""
Test runner tool: run Python tests and report results.

Provides a tool to run pytest or unittest on a file or directory.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "run_tests",
        "description": "Run Python tests using pytest or unittest. Returns test results with pass/fail status and output. Useful for verifying code changes work correctly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "File or directory path to test",
                },
                "test_type": {
                    "type": "string",
                    "enum": ["pytest", "unittest", "doctest"],
                    "description": "Type of test runner to use",
                    "default": "pytest",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Show detailed test output",
                    "default": True,
                },
            },
            "required": ["target"],
        },
    }


def _run_pytest(target: str, verbose: bool) -> dict:
    """Run pytest and return results."""
    cmd = [sys.executable, "-m", "pytest", target]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "-x"])  # Short traceback, stop on first failure
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd(),
        )
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Test execution timed out (60s)",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def _run_unittest(target: str, verbose: bool) -> dict:
    """Run unittest and return results."""
    cmd = [sys.executable, "-m", "unittest"]
    if verbose:
        cmd.append("-v")
    
    # Convert file path to module path if needed
    if target.endswith('.py'):
        # Try to run as a script
        cmd = [sys.executable, target]
    else:
        cmd.append(target)
    
    cmd.append("2>&1")  # Capture stderr too
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            shell=True,
        )
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Test execution timed out (60s)",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def _run_doctest(target: str, verbose: bool) -> dict:
    """Run doctest on a Python file."""
    cmd = [sys.executable, "-m", "doctest"]
    if verbose:
        cmd.append("-v")
    cmd.append(target)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Doctest execution timed out (60s)",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def tool_function(target: str, test_type: str = "pytest", verbose: bool = True) -> str:
    """Run tests and return formatted results."""
    if not os.path.exists(target):
        return f"Error: Target path does not exist: {target}"
    
    # Run appropriate test type
    if test_type == "pytest":
        result = _run_pytest(target, verbose)
    elif test_type == "unittest":
        result = _run_unittest(target, verbose)
    elif test_type == "doctest":
        result = _run_doctest(target, verbose)
    else:
        return f"Error: Unknown test type: {test_type}"
    
    # Format output
    lines = [f"=== Test Results ({test_type}) for {target} ==="]
    
    if "error" in result:
        lines.append(f"\n❌ Error: {result['error']}")
        return '\n'.join(lines)
    
    if result["success"]:
        lines.append("\n✅ All tests passed!")
    else:
        lines.append(f"\n❌ Tests failed (exit code: {result['returncode']})")
    
    if result.get("stdout"):
        lines.append("\n--- STDOUT ---")
        lines.append(result["stdout"][:2000])  # Limit output
    
    if result.get("stderr"):
        lines.append("\n--- STDERR ---")
        lines.append(result["stderr"][:1000])  # Limit output
    
    return '\n'.join(lines)
