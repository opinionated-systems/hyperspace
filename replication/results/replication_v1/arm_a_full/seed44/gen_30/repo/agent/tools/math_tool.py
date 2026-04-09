"""
Math verification tool: evaluate mathematical expressions and verify calculations.

Provides a safe way to check mathematical computations during grading.
"""

from __future__ import annotations

import ast
import operator
import re


def tool_info() -> dict:
    return {
        "name": "math_verify",
        "description": (
            "Evaluate and verify mathematical expressions. "
            "Supports basic arithmetic, powers, and common math functions. "
            "Use this to check if student calculations are correct."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2**100 % 3', '5 + 3 * 2').",
                },
                "expected": {
                    "type": "string",
                    "description": "Optional expected result to compare against.",
                },
            },
            "required": ["expression"],
        },
    }


# Safe operators for math evaluation
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Safely evaluate an AST node."""
    if isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.Constant):  # Python >= 3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](operand)
    elif isinstance(node, ast.Call):
        # Allow some safe math functions
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ('abs', 'max', 'min', 'sum'):
                args = [_eval_node(arg) for arg in node.args]
                return globals()[func_name](args) if func_name == 'sum' else globals()[func_name](*args)
        raise ValueError(f"Unsupported function call")
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def _safe_eval(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    # Clean the expression
    expression = expression.strip()
    
    # Replace common math notation
    expression = expression.replace('^', '**')
    expression = expression.replace('×', '*')
    expression = expression.replace('÷', '/')
    
    # Parse and evaluate
    try:
        tree = ast.parse(expression, mode='eval')
        result = _eval_node(tree.body)
        return result
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {e}")


def tool_function(expression: str, expected: str | None = None) -> str:
    """Evaluate a mathematical expression and optionally compare with expected result.
    
    Args:
        expression: The mathematical expression to evaluate
        expected: Optional expected result to compare against
        
    Returns:
        String with the evaluation result and comparison if expected was provided
    """
    try:
        result = _safe_eval(expression)
        
        # Format result
        if result == int(result):
            result_str = str(int(result))
        else:
            result_str = f"{result:.10g}"
        
        output = f"Expression: {expression}\nResult: {result_str}"
        
        if expected is not None:
            try:
                expected_val = _safe_eval(expected)
                if result == expected_val:
                    output += f"\n✓ Matches expected value: {expected}"
                else:
                    output += f"\n✗ Does NOT match expected value: {expected}"
                    output += f"\n  Difference: {abs(result - expected_val)}"
            except Exception as e:
                output += f"\n? Could not evaluate expected value '{expected}': {e}"
        
        return output
        
    except Exception as e:
        return f"Error evaluating expression: {e}"
