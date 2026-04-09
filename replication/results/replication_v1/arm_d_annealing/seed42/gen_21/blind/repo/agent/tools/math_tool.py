"""
Math tool: provides mathematical computation capabilities.

Supports arithmetic operations, algebraic calculations, and common math functions.
"""

from __future__ import annotations

import math
import json
from typing import Any


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "math",
        "description": "Perform mathematical calculations including arithmetic, algebra, and common math functions like sqrt, sin, cos, log, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')",
                },
                "variables": {
                    "type": "object",
                    "description": "Optional dictionary of variable names and values to use in the expression",
                    "default": {},
                },
            },
            "required": ["expression"],
        },
    }


def tool_function(expression: str, variables: dict | None = None) -> str:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression as a string
        variables: Optional dict of variable values
        
    Returns:
        Result as a string, or error message
    """
    if variables is None:
        variables = {}
    
    # Define safe math functions and constants
    safe_dict = {
        # Constants
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        "nan": math.nan,
        # Basic arithmetic
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        "sum": sum,
        "pow": pow,
        # Math functions
        "sqrt": math.sqrt,
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
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "degrees": math.degrees,
        "radians": math.radians,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "isclose": math.isclose,
    }
    
    # Add user-provided variables
    safe_dict.update(variables)
    
    try:
        # Evaluate the expression in the safe environment
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return json.dumps({"result": result, "success": True})
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})
