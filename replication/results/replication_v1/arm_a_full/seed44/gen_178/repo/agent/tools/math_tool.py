"""
Math tool: perform mathematical calculations and evaluations.

Provides safe mathematical operations for evaluating expressions,
useful for verifying calculations in student answers.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

# Allowed operators for safe evaluation
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "max": max,
    "min": min,
    "round": round,
    "sum": sum,
    "pow": pow,
}

# Mathematical constants
_ALLOWED_NAMES = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
    "inf": float("inf"),
    "nan": float("nan"),
}


def _safe_eval(node: ast.AST) -> Any:
    """Safely evaluate an AST node."""
    if isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.Constant):  # Python >= 3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [_safe_eval(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise ValueError(f"Unknown name: {node.id}")
        return _ALLOWED_NAMES[node.id]
    elif isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def evaluate_expression(expression: str) -> str:
    """Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "3 * 4", "2**10")
        
    Returns:
        Result as a string, or error message if evaluation fails
    """
    try:
        # Parse the expression
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "math",
        "description": "Evaluate mathematical expressions safely. Supports basic arithmetic (+, -, *, /, **), floor division (//), modulo (%), and functions like abs(), max(), min(), round(), sum(), pow(). Also supports constants pi, e, inf, nan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '3.14 * r**2', 'abs(-5)')",
                },
            },
            "required": ["expression"],
        },
    }


def tool_function(expression: str) -> str:
    """Execute the math tool."""
    return evaluate_expression(expression)
