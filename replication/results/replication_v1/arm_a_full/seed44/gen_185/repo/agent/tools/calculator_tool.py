"""
Calculator tool: performs mathematical calculations.

Provides a simple interface for evaluating mathematical expressions.
Supports basic arithmetic, scientific functions, and common operations.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any


def calculator(expression: str) -> dict[str, Any]:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression as a string.
                   Examples: "2 + 2", "sqrt(16)", "sin(pi/2)", "10!"

    Returns:
        dict with 'result' (float or int) and 'error' (str or None)
    """
    try:
        result = _safe_eval(expression)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _safe_eval(expression: str) -> float | int:
    """Safely evaluate a mathematical expression.
    
    Uses AST parsing to only allow safe mathematical operations.
    """
    # Define allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Define allowed functions and constants
    allowed_names = {
        # Math functions
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'pow': math.pow,
        'factorial': math.factorial,
        'floor': math.floor,
        'ceil': math.ceil,
        'abs': abs,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        # Constants
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
    }
    
    def eval_node(node: ast.AST) -> float | int:
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in allowed_names:
                raise ValueError(f"Unknown function: {func_name}")
            func = allowed_names[func_name]
            args = [eval_node(arg) for arg in node.args]
            return func(*args)
        elif isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"Unknown name: {node.id}")
            return allowed_names[node.id]
        elif isinstance(node, ast.Expression):
            return eval_node(node.body)
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")
    
    # Parse and evaluate
    try:
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree)
        # Convert to int if it's a whole number
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    except Exception as e:
        raise ValueError(f"Evaluation error: {e}")


def tool_info() -> dict:
    return {
        "name": "calculator",
        "description": (
            "Evaluate mathematical expressions safely. "
            "Supports basic arithmetic (+, -, *, /, //, %, **), "
            "scientific functions (sqrt, sin, cos, tan, exp, log, factorial, etc.), "
            "and constants (pi, e, tau). Trigonometric functions use radians. "
            "Examples: '2 + 3 * 4', 'sqrt(16) + pi', 'sin(pi/2)', 'factorial(5)'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate. Examples: '2 + 2', 'sqrt(16)', 'sin(pi/2)', 'factorial(5)'",
                }
            },
            "required": ["expression"],
        },
    }


def tool_function(expression: str) -> dict:
    """Main entry point for the tool."""
    return calculator(expression)
