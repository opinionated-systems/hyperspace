"""
Calculator tool for mathematical operations.

Provides basic arithmetic and mathematical functions useful for
verifying calculations in IMO grading problems.
"""

from __future__ import annotations

import math
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "calculator",
        "description": "Perform mathematical calculations including arithmetic, powers, roots, logarithms, and trigonometric functions. Useful for verifying numerical results in grading problems.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate. Supports: +, -, *, /, ** (power), sqrt, log, ln, sin, cos, tan, abs, pi, e. Example: '2**10 + sqrt(16)'",
                },
                "precision": {
                    "type": "integer",
                    "description": "Number of decimal places to round to (default: 6).",
                    "default": 6,
                },
            },
            "required": ["expression"],
        },
    }


def tool_function(expression: str, precision: int = 6) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Define safe math functions and constants
        safe_dict = {
            # Constants
            "pi": math.pi,
            "e": math.e,
            # Basic functions
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            # Math functions
            "sqrt": math.sqrt,
            "log": math.log10,
            "ln": math.log,
            "log2": math.log2,
            "exp": math.exp,
            "pow": math.pow,
            # Trigonometric functions
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            # Other functions
            "floor": math.floor,
            "ceil": math.ceil,
            "factorial": math.factorial,
            "gcd": math.gcd,
            "degrees": math.degrees,
            "radians": math.radians,
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        # Round to specified precision
        if isinstance(result, float):
            result = round(result, precision)
            # Remove trailing zeros
            result = str(result).rstrip('0').rstrip('.') if '.' in str(result) else str(result)
        
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid value - {e}"
    except OverflowError:
        return "Error: Result too large"
    except Exception as e:
        return f"Error evaluating expression: {e}"
