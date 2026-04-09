"""
Math tool: perform mathematical calculations and operations.

Provides safe evaluation of mathematical expressions and common math utilities.
"""

from __future__ import annotations

import math
import operator
import re
from typing import Any


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "math",
        "description": "Perform mathematical calculations and operations. Supports basic arithmetic (+, -, *, /, **, %), mathematical functions (sqrt, sin, cos, tan, log, exp, etc.), and constants (pi, e).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)', 'log(100, 10)')",
                },
                "operation": {
                    "type": "string",
                    "description": "Named operation to perform (alternative to expression). Options: add, subtract, multiply, divide, power, sqrt, abs, round, floor, ceil, sin, cos, tan, log, log10, exp, factorial, gcd, lcm",
                },
                "args": {
                    "type": "array",
                    "description": "Arguments for named operation",
                    "items": {"type": "number"},
                },
            },
            "required": [],
        },
    }


# Safe math environment
SAFE_MATH = {
    # Constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    # Basic operations
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "sum": sum,
    # Math module functions
    "sqrt": math.sqrt,
    "pow": math.pow,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm if hasattr(math, "lcm") else lambda a, b: abs(a * b) // math.gcd(a, b) if a and b else 0,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
    "hypot": math.hypot,
    "dist": math.dist if hasattr(math, "dist") else lambda p, q: math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q))),
    "comb": math.comb if hasattr(math, "comb") else None,
    "perm": math.perm if hasattr(math, "perm") else None,
}

# Add None-check for comb and perm
if SAFE_MATH["comb"] is None:
    def _comb(n, k):
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c
    SAFE_MATH["comb"] = _comb

if SAFE_MATH["perm"] is None:
    def _perm(n, k):
        if k < 0 or k > n:
            return 0
        result = 1
        for i in range(k):
            result *= (n - i)
        return result
    SAFE_MATH["perm"] = _perm


def _safe_eval(expression: str) -> Any:
    """Safely evaluate a mathematical expression."""
    # Clean the expression
    expression = expression.strip()
    if not expression:
        raise ValueError("Empty expression")
    
    # Remove any potentially dangerous characters/patterns
    # Only allow: digits, operators, parentheses, dots, commas, spaces, letters for function names
    allowed_pattern = re.compile(r'^[\d\+\-\*\/\%\(\)\.\,\s\_a-zA-Z]+$')
    if not allowed_pattern.match(expression):
        raise ValueError(f"Expression contains invalid characters: {expression}")
    
    # Replace power operator
    expression = expression.replace("**", "^")  # Temp marker
    
    # Tokenize and validate
    tokens = re.findall(r'\d+\.?\d*|[a-zA-Z_]+|\+|\-|\*|\/|\%|\(|\)|\^|\,', expression)
    
    # Rebuild with safe replacements
    safe_tokens = []
    for token in tokens:
        if token == "^":
            safe_tokens.append("**")
        elif token in SAFE_MATH or token in ("+", "-", "*", "/", "%", "(", ")", ","):
            safe_tokens.append(token)
        elif re.match(r'\d+\.?\d*$', token):
            safe_tokens.append(token)
        else:
            raise ValueError(f"Unknown token: {token}")
    
    safe_expr = " ".join(safe_tokens)
    
    # Evaluate with limited globals and locals
    result = eval(safe_expr, {"__builtins__": {}}, SAFE_MATH)
    return result


def tool_function(expression: str | None = None, operation: str | None = None, args: list | None = None) -> str:
    """Execute math operation or evaluate expression."""
    try:
        if expression is not None:
            # Evaluate expression
            result = _safe_eval(expression)
            return f"Result: {result}"
        
        elif operation is not None and args is not None:
            # Named operation
            op = operation.lower()
            
            if op == "add":
                result = sum(args)
            elif op == "subtract":
                result = args[0] - sum(args[1:]) if len(args) > 1 else args[0]
            elif op == "multiply":
                result = 1
                for a in args:
                    result *= a
            elif op == "divide":
                result = args[0]
                for a in args[1:]:
                    result /= a
            elif op == "power":
                result = args[0] ** args[1] if len(args) >= 2 else args[0]
            elif op == "sqrt":
                result = math.sqrt(args[0])
            elif op == "abs":
                result = abs(args[0])
            elif op == "round":
                result = round(args[0], int(args[1]) if len(args) > 1 else 0)
            elif op == "floor":
                result = math.floor(args[0])
            elif op == "ceil":
                result = math.ceil(args[0])
            elif op == "sin":
                result = math.sin(args[0])
            elif op == "cos":
                result = math.cos(args[0])
            elif op == "tan":
                result = math.tan(args[0])
            elif op == "log":
                result = math.log(args[0], args[1]) if len(args) > 1 else math.log(args[0])
            elif op == "log10":
                result = math.log10(args[0])
            elif op == "exp":
                result = math.exp(args[0])
            elif op == "factorial":
                result = math.factorial(int(args[0]))
            elif op == "gcd":
                result = math.gcd(int(args[0]), int(args[1])) if len(args) >= 2 else args[0]
            elif op == "lcm":
                a, b = int(args[0]), int(args[1])
                result = math.lcm(a, b) if hasattr(math, "lcm") else abs(a * b) // math.gcd(a, b)
            else:
                return f"Error: Unknown operation '{operation}'"
            
            return f"Result: {result}"
        
        else:
            return "Error: Provide either 'expression' or both 'operation' and 'args'"
    
    except Exception as e:
        return f"Error: {e}"
