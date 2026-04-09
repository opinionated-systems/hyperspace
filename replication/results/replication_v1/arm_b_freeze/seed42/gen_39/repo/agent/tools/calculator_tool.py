"""
Calculator tool: perform mathematical calculations.

Provides basic arithmetic operations and mathematical functions.
"""

from __future__ import annotations

import math
import json
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool specification for LLM."""
    return {
        "name": "calculator",
        "description": "Perform mathematical calculations. Supports basic arithmetic (+, -, *, /), powers, square roots, trigonometric functions, logarithms, and constants like pi and e.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)', 'log(100, 10)')",
                },
            },
            "required": ["expression"],
        },
    }


def tool_function(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression as string
        
    Returns:
        Result as string or error message
    """
    try:
        # Create safe evaluation environment
        safe_dict = {
            # Basic math
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
            # Trigonometry
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "degrees": math.degrees,
            "radians": math.radians,
            # Constants
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
            "nan": math.nan,
            # Rounding
            "ceil": math.ceil,
            "floor": math.floor,
            "factorial": math.factorial,
        }
        
        # Evaluate expression safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        # Format result
        if isinstance(result, float):
            # Round to avoid floating point artifacts
            if abs(result - round(result)) < 1e-10:
                return str(int(round(result)))
            return f"{result:.10f}".rstrip("0").rstrip(".")
        
        return str(result)
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid value - {e}"
    except OverflowError:
        return "Error: Result too large"
    except Exception as e:
        return f"Error: Could not evaluate expression - {e}"
