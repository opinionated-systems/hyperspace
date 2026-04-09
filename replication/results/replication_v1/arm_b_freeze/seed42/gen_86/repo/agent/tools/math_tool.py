"""
Math tool: provides mathematical utilities for IMO grading problems.

Includes functions for symbolic computation, equation solving, and
mathematical expression evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any


def tool_info() -> dict:
    """Return tool metadata for LLM."""
    return {
        "name": "math",
        "description": "Mathematical utilities for symbolic computation and expression evaluation. Supports equation solving, expression simplification, and numerical computation.",
        "functions": [
            {
                "name": "evaluate_expression",
                "description": "Evaluate a mathematical expression and return the result",
                "parameters": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
                    }
                },
                "required": ["expression"]
            },
            {
                "name": "solve_linear",
                "description": "Solve a linear equation for a variable",
                "parameters": {
                    "equation": {
                        "type": "string",
                        "description": "Linear equation to solve (e.g., '2x + 3 = 7')"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to solve for (default: 'x')"
                    }
                },
                "required": ["equation"]
            },
            {
                "name": "check_equality",
                "description": "Check if two mathematical expressions are equal",
                "parameters": {
                    "expr1": {
                        "type": "string",
                        "description": "First expression"
                    },
                    "expr2": {
                        "type": "string",
                        "description": "Second expression"
                    }
                },
                "required": ["expr1", "expr2"]
            }
        ]
    }


def _safe_eval(expression: str) -> Any:
    """Safely evaluate a mathematical expression."""
    # Basic sanitization - only allow safe characters
    allowed_chars = set("0123456789+-*/.()^ sqrtabsloglnexpipi ")
    if not all(c in allowed_chars for c in expression.lower()):
        raise ValueError(f"Expression contains invalid characters: {expression}")
    
    # Replace common mathematical notation
    expr = expression.lower()
    expr = expr.replace("^", "**")
    expr = expr.replace("sqrt", "__import__('math').sqrt")
    expr = expr.replace("abs", "__import__('math').fabs")
    expr = expr.replace("log", "__import__('math').log10")
    expr = expr.replace("ln", "__import__('math').log")
    expr = expr.replace("exp", "__import__('math').exp")
    expr = expr.replace("pi", str(__import__('math').pi))
    
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return result
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}")


def _solve_linear(equation: str, variable: str = "x") -> str:
    """Solve a simple linear equation."""
    # Parse equation of form ax + b = c
    equation = equation.replace(" ", "")
    if "=" not in equation:
        raise ValueError("Equation must contain '='")
    
    left, right = equation.split("=", 1)
    
    # Extract coefficient and constant from left side
    # Pattern: ax + b or ax - b or x + b or x - b or ax or x
    pattern = rf'([+-]?\d*?){re.escape(variable)}([+-]\d+)?$'
    match = re.match(pattern, left)
    
    if not match and left == variable:
        a = 1
        b = 0
    elif match:
        a_str = match.group(1)
        a = int(a_str) if a_str and a_str not in ['+', '-'] else (1 if a_str != '-' else -1)
        b_str = match.group(2)
        b = int(b_str) if b_str else 0
    else:
        raise ValueError(f"Could not parse equation: {equation}")
    
    # Evaluate right side
    c = _safe_eval(right)
    
    # Solve: ax + b = c => x = (c - b) / a
    if a == 0:
        raise ValueError("Coefficient of variable cannot be zero")
    
    result = (c - b) / a
    return str(result)


def tool_function(function_name: str, **kwargs) -> str:
    """Execute a math function."""
    try:
        if function_name == "evaluate_expression":
            expression = kwargs.get("expression", "")
            result = _safe_eval(expression)
            return json.dumps({"result": result, "success": True})
        
        elif function_name == "solve_linear":
            equation = kwargs.get("equation", "")
            variable = kwargs.get("variable", "x")
            result = _solve_linear(equation, variable)
            return json.dumps({"result": result, "success": True})
        
        elif function_name == "check_equality":
            expr1 = kwargs.get("expr1", "")
            expr2 = kwargs.get("expr2", "")
            val1 = _safe_eval(expr1)
            val2 = _safe_eval(expr2)
            # Use approximate equality for floating point
            equal = abs(val1 - val2) < 1e-9
            return json.dumps({"equal": equal, "value1": val1, "value2": val2, "success": True})
        
        else:
            return json.dumps({"error": f"Unknown function: {function_name}", "success": False})
    
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})
