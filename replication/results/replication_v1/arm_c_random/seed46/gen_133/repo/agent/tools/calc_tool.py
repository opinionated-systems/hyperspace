"""
Calculator tool: perform mathematical calculations.

Provides safe mathematical evaluation for the agent.
Supports basic arithmetic, scientific functions, and common operations.
"""

from __future__ import annotations

import math
from typing import Any


def tool_info() -> dict:
    return {
        "name": "calc",
        "description": (
            "Perform mathematical calculations safely. "
            "Supports arithmetic (+, -, *, /, **), math functions (sqrt, sin, cos, log, etc.), "
            "and constants (pi, e). "
            "Use this for any numerical computations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)').",
                },
            },
            "required": ["expression"],
        },
    }


# Safe math environment
_SAFE_MATH = {
    # Constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    # Basic functions
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "sum": sum,
    "pow": pow,
    # Math module functions
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
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
    "hypot": math.hypot,
    "dist": math.dist,
    "isclose": math.isclose,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
}


def tool_function(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: The mathematical expression to evaluate.
        
    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    try:
        # Compile and evaluate with restricted globals
        code = compile(expression, "<string>", "eval")
        result = eval(code, {"__builtins__": {}}, _SAFE_MATH)
        
        # Format result nicely
        if isinstance(result, float):
            # Handle very small or very large numbers
            if abs(result) < 1e-10 or abs(result) > 1e10:
                return f"{result:.6e}"
            # Round to reasonable precision
            return f"{result:.10f}".rstrip("0").rstrip(".")
        elif isinstance(result, int):
            return str(result)
        else:
            return str(result)
    except SyntaxError as e:
        return f"Error: Invalid syntax in expression '{expression}': {e}"
    except NameError as e:
        return f"Error: Unknown name in expression '{expression}': {e}"
    except ZeroDivisionError:
        return f"Error: Division by zero in expression '{expression}'"
    except OverflowError:
        return f"Error: Numeric overflow in expression '{expression}'"
    except Exception as e:
        return f"Error evaluating '{expression}': {type(e).__name__}: {e}"
