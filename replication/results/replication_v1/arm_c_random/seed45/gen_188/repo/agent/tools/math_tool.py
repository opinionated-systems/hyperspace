"""
Math tool: perform mathematical calculations and evaluations.

Provides safe mathematical computation capabilities for the agent,
including expression evaluation, number theory operations, and
statistical calculations useful for IMO grading verification.
"""

from __future__ import annotations

import math
import operator
import re
from typing import Any


# Safe math operations whitelist
_SAFE_MATH = {
    # Basic operations
    'abs': abs,
    'round': round,
    'max': max,
    'min': min,
    'sum': sum,
    'len': len,
    'pow': pow,
    'divmod': divmod,
    
    # Math module functions
    'sqrt': math.sqrt,
    'floor': math.floor,
    'ceil': math.ceil,
    'factorial': math.factorial,
    'gcd': math.gcd,
    'lcm': math.lcm if hasattr(math, 'lcm') else lambda a, b: abs(a * b) // math.gcd(a, b),
    'isqrt': math.isqrt,
    'log': math.log,
    'log10': math.log10,
    'log2': math.log2,
    'exp': math.exp,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'pi': math.pi,
    'e': math.e,
    'inf': math.inf,
    'nan': math.nan,
    'isfinite': math.isfinite,
    'isinf': math.isinf,
    'isnan': math.isnan,
    
    # Number theory
    'mod': operator.mod,
    'floordiv': operator.floordiv,
}


def tool_info() -> dict:
    return {
        "name": "math",
        "description": (
            "Perform mathematical calculations and evaluations safely. "
            "Supports arithmetic, number theory, and basic statistical operations. "
            "Useful for verifying calculations in student answers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2**100 % 3', 'gcd(48, 18)', 'sqrt(2)').",
                },
                "operation": {
                    "type": "string",
                    "enum": ["evaluate", "factor", "is_prime", "mod_exp"],
                    "description": "Type of mathematical operation to perform.",
                    "default": "evaluate",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers for operations that need multiple values (e.g., gcd, lcm).",
                },
            },
            "required": ["expression"],
        },
    }


def _safe_eval(expression: str) -> Any:
    """Safely evaluate a mathematical expression.
    
    Only allows whitelisted math operations and numeric literals.
    """
    # Clean the expression
    expression = expression.strip()
    
    # Remove any potentially dangerous characters/patterns
    # Block any attempts to access attributes or call non-whitelisted functions
    dangerous_patterns = [
        r'__',  # Dunder methods
        r'import\s',
        r'from\s',
        r'exec\s*\(',
        r'eval\s*\(',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'os\.',
        r'sys\.',
        r'subprocess\.',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            raise ValueError(f"Expression contains dangerous pattern: {pattern}")
    
    # Parse and evaluate with limited globals
    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        return result
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")


def _is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _factor(n: int) -> list[tuple[int, int]]:
    """Return prime factorization as list of (prime, power) tuples."""
    if n < 2:
        return []
    
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count > 0:
            factors.append((d, count))
        d += 1 if d == 2 else 2  # Skip even numbers after 2
    
    if n > 1:
        factors.append((n, 1))
    
    return factors


def _mod_exp(base: int, exp: int, mod: int) -> int:
    """Compute (base^exp) % mod efficiently using modular exponentiation."""
    return pow(base, exp, mod)


def tool_function(
    expression: str,
    operation: str = "evaluate",
    values: list[float] | None = None,
) -> str:
    """Execute a mathematical operation.
    
    Args:
        expression: Mathematical expression or number to operate on
        operation: Type of operation (evaluate, factor, is_prime, mod_exp)
        values: Additional values for multi-value operations
    
    Returns:
        Result of the mathematical operation as a string
    """
    try:
        if operation == "evaluate":
            result = _safe_eval(expression)
            return f"Result: {result}"
        
        elif operation == "factor":
            n = int(_safe_eval(expression))
            if n < 2:
                return f"{n} has no prime factors (must be >= 2)"
            factors = _factor(n)
            factor_str = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in factors)
            return f"Prime factorization of {n}: {factor_str} = {n}"
        
        elif operation == "is_prime":
            n = int(_safe_eval(expression))
            is_p = _is_prime(n)
            return f"{n} is {'prime' if is_p else 'not prime'}"
        
        elif operation == "mod_exp":
            # Parse expression as "base,exp,mod"
            parts = expression.split(",")
            if len(parts) != 3:
                return "Error: mod_exp requires 'base,exp,mod' format"
            base, exp, mod = map(int, parts)
            result = _mod_exp(base, exp, mod)
            return f"{base}^{exp} mod {mod} = {result}"
        
        elif operation == "gcd":
            if values and len(values) >= 2:
                result = math.gcd(int(values[0]), int(values[1]))
                return f"gcd({values[0]}, {values[1]}) = {result}"
            return "Error: gcd requires at least 2 values"
        
        elif operation == "lcm":
            if values and len(values) >= 2:
                a, b = int(values[0]), int(values[1])
                result = abs(a * b) // math.gcd(a, b)
                return f"lcm({a}, {b}) = {result}"
            return "Error: lcm requires at least 2 values"
        
        else:
            return f"Error: Unknown operation '{operation}'"
            
    except Exception as e:
        return f"Error: {e}"
