"""
Calculator tool: perform mathematical calculations.

Provides basic arithmetic operations and mathematical functions
to help the agent with numerical computations.
"""

from __future__ import annotations

import math
from typing import Any


def tool_info() -> dict[str, Any]:
    return {
        "name": "calculator",
        "description": "Perform mathematical calculations. Supports basic arithmetic (+, -, *, /, **), mathematical functions (sqrt, sin, cos, tan, log, exp), and constants (pi, e).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate. Use Python syntax. Examples: '2 + 2', 'sqrt(16)', 'sin(pi/2)', 'log(100, 10)'",
                },
            },
            "required": ["expression"],
        },
    }


def tool_function(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Define safe math functions and constants
        safe_dict = {
            # Basic arithmetic
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            # Math functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "pow": math.pow,
            "factorial": math.factorial,
            "floor": math.floor,
            "ceil": math.ceil,
            "trunc": math.trunc,
            "gcd": math.gcd,
            "lcm": math.lcm if hasattr(math, "lcm") else None,
            "isclose": math.isclose,
            # Constants
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau if hasattr(math, "tau") else 2 * math.pi,
            "inf": math.inf,
            "nan": math.nan,
        }
        
        # Remove None values (for Python version compatibility)
        safe_dict = {k: v for k, v in safe_dict.items() if v is not None}
        
        # Compile and evaluate the expression
        code = compile(expression, "<string>", "eval")
        
        # Check that only allowed names are used
        for name in code.co_names:
            if name not in safe_dict and name not in ("__builtins__",):
                return f"Error: '{name}' is not allowed in calculator expressions"
        
        result = eval(code, {"__builtins__": {}}, safe_dict)
        
        # Format the result nicely
        if isinstance(result, float):
            # Handle very small or very large numbers
            if abs(result) < 0.0001 or abs(result) > 1000000:
                return f"{result:.6e}"
            # Round to reasonable precision
            return f"{result:.10f}".rstrip("0").rstrip(".")
        elif isinstance(result, int):
            return str(result)
        else:
            return str(result)
            
    except SyntaxError as e:
        return f"Error: Invalid syntax in expression '{expression}': {e}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except OverflowError:
        return "Error: Numeric overflow"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error evaluating expression: {e}"
