"""
Math tool: perform mathematical calculations.

Provides basic arithmetic and mathematical operations for agents.
"""

from __future__ import annotations

import math
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "math",
        "description": "Perform mathematical calculations including arithmetic, powers, roots, trigonometry, and logarithms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)'). Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil, pi, e",
                },
            },
            "required": ["expression"],
        },
    }


def tool_function(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: The mathematical expression to evaluate

    Returns:
        The result of the calculation as a string
    """
    try:
        # Create a safe evaluation environment with math functions
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "abs": abs,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
            "pi": math.pi,
            "e": math.e,
            "pow": pow,
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        return f"Result: {result}"
    
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid value - {e}"
    except SyntaxError:
        return f"Error: Invalid syntax in expression: {expression}"
    except Exception as e:
        return f"Error: {e}"
