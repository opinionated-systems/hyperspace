"""
Test tool: run Python tests and check code quality.

Provides pytest integration and basic code quality checks
like linting and type checking. Useful for validating
modifications to the codebase.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "test",
        "description": (
            "Run Python tests and code quality checks. "
            "Supports pytest for testing, ruff for linting, "
            "and mypy for type checking. Useful for validating "
            "code changes work correctly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["pytest", "lint", "typecheck"],
                    "description": "The test command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to test (file or directory). Default: current directory.",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Show verbose output (default: False).",
                },
            },
            "required": ["command"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for test operations."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_root(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    try:
        Path(path).resolve().relative_to(Path(_ALLOWED_ROOT).resolve())
        return True
    except ValueError:
        return False


def _truncate_output(output: str, max_len: int = 10000) -> str:
    """Truncate output if it exceeds max length."""
    if len(output) <= max_len:
        return output
    
    half = max_len // 2
    start = output[:half]
    end = output[-half:]
    truncated_len = len(output) - max_len
    return f"{start}\n\n... [{truncated_len} characters truncated] ...\n\n{end}"


def tool_function(
    command: str,
    path: str | None = None,
    verbose: bool = False,
) -> str:
    """Run tests or code quality checks.

    Args:
        command: The test command ('pytest', 'lint', or 'typecheck')
        path: Path to test (file or directory). Default: allowed root or current dir
        verbose: Show verbose output

    Returns:
        Test results with summary and any errors found
    """
    target_path = path or _ALLOWED_ROOT or "."
    
    if not _is_within_root(target_path):
        return f"Error: Path '{target_path}' is outside allowed root."

    if not Path(target_path).exists():
        return f"Error: Path '{target_path}' does not exist."

    try:
        if command == "pytest":
            return _run_pytest(target_path, verbose)
        elif command == "lint":
            return _run_lint(target_path, verbose)
        elif command == "typecheck":
            return _run_typecheck(target_path, verbose)
        else:
            return f"Error: Unknown command '{command}'. Use 'pytest', 'lint', or 'typecheck'."
    except Exception as e:
        return f"Error: {e}"


def _run_pytest(path: str, verbose: bool) -> str:
    """Run pytest on the given path."""
    cmd = ["python", "-m", "pytest", path]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "-q"])
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    
    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    
    # Add summary
    if result.returncode == 0:
        summary = "\n✓ Tests passed"
    else:
        summary = f"\n✗ Tests failed (exit code: {result.returncode})"
    
    return _truncate_output(output + summary)


def _run_lint(path: str, verbose: bool) -> str:
    """Run ruff linter on the given path."""
    # First try ruff, fall back to flake8
    for linter, cmd_base in [("ruff", ["ruff", "check"]), ("flake8", ["flake8"])]:
        try:
            cmd = cmd_base + [path]
            if verbose:
                cmd.append("-v")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            
            if result.returncode == 0:
                summary = f"\n✓ {linter} check passed - no issues found"
            else:
                summary = f"\n✗ {linter} found issues (exit code: {result.returncode})"
            
            return _truncate_output(output + summary)
        except FileNotFoundError:
            continue
    
    return "Error: No linter found (tried ruff, flake8). Please install one."


def _run_typecheck(path: str, verbose: bool) -> str:
    """Run mypy type checker on the given path."""
    cmd = ["python", "-m", "mypy", path]
    if verbose:
        cmd.append("-v")
    cmd.append("--ignore-missing-imports")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    
    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    
    if result.returncode == 0:
        summary = "\n✓ Type check passed - no issues found"
    else:
        summary = f"\n✗ Type check found issues (exit code: {result.returncode})"
    
    return _truncate_output(output + summary)
