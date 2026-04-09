"""
Syntax validator tool: quick Python syntax validation without full analysis.

Provides fast syntax checking for Python code, useful for validating
changes before applying them. Much faster than full code analysis.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "syntax_validator",
        "description": (
            "Quickly validate Python syntax without full code analysis. "
            "Checks if Python code is syntactically valid. "
            "Useful for validating code changes before applying them. "
            "Much faster than code_analyzer for syntax-only checks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to Python file to validate.",
                },
                "code": {
                    "type": "string",
                    "description": "Python code string to validate (alternative to path).",
                },
            },
            "oneOf": [
                {"required": ["path"]},
                {"required": ["code"]},
            ],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict validation operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Validation restricted to {_ALLOWED_ROOT}"
    return True, ""


def tool_function(
    path: str | None = None,
    code: str | None = None,
) -> str:
    """Validate Python syntax for a file or code string."""
    if path is not None and code is not None:
        return "Error: Provide either 'path' or 'code', not both."
    
    if path is None and code is None:
        return "Error: Provide either 'path' or 'code'."
    
    if path is not None:
        return _validate_file(path)
    else:
        return _validate_code(code or "", "<string>")


def _validate_file(path: str) -> str:
    """Validate a Python file."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if not p.is_file():
            return f"Error: {p} is not a file."
        
        if p.suffix != ".py":
            return f"Error: {p} is not a Python file (.py)."
        
        try:
            source = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"Error reading file: {e}"
        
        return _validate_code(source, str(p))
        
    except Exception as e:
        return f"Error: {e}"


def _validate_code(source: str, filename: str) -> str:
    """Validate Python code string."""
    if not source.strip():
        return f"✓ {filename}: Empty file is valid (no code to validate)"
    
    try:
        ast.parse(source, filename=filename)
        lines = source.strip().split('\n')
        line_count = len(lines)
        non_empty = len([l for l in lines if l.strip()])
        return f"✓ {filename}: Syntax OK ({non_empty} lines of code, {line_count} total lines)"
    except SyntaxError as e:
        error_msg = f"✗ {filename}: Syntax Error"
        if e.lineno:
            error_msg += f" at line {e.lineno}"
            if e.offset:
                error_msg += f", column {e.offset}"
        error_msg += f": {e.msg}"
        
        # Show the problematic line if available
        if e.lineno and e.lineno > 0:
            lines = source.split('\n')
            if e.lineno <= len(lines):
                line = lines[e.lineno - 1]
                error_msg += f"\n  Line {e.lineno}: {line[:80]}"
                if e.offset and e.offset > 0:
                    pointer = " " * (e.offset + 10) + "^"
                    error_msg += f"\n  {pointer}"
        
        return error_msg
    except IndentationError as e:
        error_msg = f"✗ {filename}: Indentation Error"
        if e.lineno:
            error_msg += f" at line {e.lineno}"
        error_msg += f": {e.msg}"
        return error_msg
    except Exception as e:
        return f"✗ {filename}: Parse Error: {type(e).__name__}: {e}"
