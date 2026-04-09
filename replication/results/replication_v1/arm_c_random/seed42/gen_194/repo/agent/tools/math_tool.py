"""
Math tool: perform mathematical calculations and symbolic operations.

Provides safe evaluation of mathematical expressions and symbolic computation.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any


def _safe_eval(expr: str) -> float:
    """Safely evaluate a mathematical expression.
    
    Only allows basic arithmetic operations and numeric literals.
    """
    # Whitelist of allowed node types
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )
    
    # Parse the expression
    try:
        tree = ast.parse(expr.strip(), mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    
    # Validate all nodes are allowed
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed operation in expression: {type(node).__name__}")
    
    # Evaluate safely
    def eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Non-numeric constant: {node.value}")
        elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            elif isinstance(node.op, ast.Pow):
                return left ** right
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.FloorDiv):
                return left // right
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
        raise ValueError(f"Unsupported node type: {type(node).__name__}")
    
    return eval_node(tree.body)


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "math",
        "description": "Perform mathematical calculations safely. Supports basic arithmetic (+, -, *, /, **, %, //), parentheses, and numeric literals. Use this to verify calculations or compute values when grading mathematical problems.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate. Examples: '2 + 2', '(10 - 3) * 5', '2**10', '100 / 4'",
                },
                "operation": {
                    "type": "string",
                    "enum": ["evaluate", "compare"],
                    "description": "The operation to perform. 'evaluate' computes the expression value. 'compare' checks if two expressions are equal.",
                },
                "expected": {
                    "type": "string",
                    "description": "For 'compare' operation: the expected value or expression to compare against.",
                },
            },
            "required": ["expression", "operation"],
        },
    }


def tool_function(expression: str, operation: str = "evaluate", expected: str = "") -> str:
    """Execute the math tool.
    
    Args:
        expression: The mathematical expression to evaluate
        operation: Either 'evaluate' or 'compare'
        expected: The expected value for comparison (when operation='compare')
    
    Returns:
        JSON string with the result
    """
    import json
    
    try:
        # Clean the expression
        cleaned = expression.strip()
        
        if operation == "evaluate":
            result = _safe_eval(cleaned)
            return json.dumps({
                "success": True,
                "expression": expression,
                "result": result,
                "result_str": str(result),
            })
        
        elif operation == "compare":
            actual = _safe_eval(cleaned)
            expected_val = _safe_eval(expected.strip())
            is_equal = abs(actual - expected_val) < 1e-9  # Floating point tolerance
            
            return json.dumps({
                "success": True,
                "expression": expression,
                "expected": expected,
                "actual": actual,
                "expected_val": expected_val,
                "equal": is_equal,
                "message": f"Values {'match' if is_equal else 'differ'}: {actual} vs {expected_val}",
            })
        
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown operation: {operation}. Use 'evaluate' or 'compare'.",
            })
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "expression": expression,
        })
