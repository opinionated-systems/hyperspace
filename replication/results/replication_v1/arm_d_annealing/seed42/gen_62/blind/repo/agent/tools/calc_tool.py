"""
Calculator tool: perform basic arithmetic operations.

Provides addition, subtraction, multiplication, division, and power operations.
"""

from __future__ import annotations

import math


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic calculations. Supports +, -, *, /, ** (power), and sqrt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt"],
                        "description": "The arithmetic operation to perform",
                    },
                    "a": {
                        "type": "number",
                        "description": "First operand",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second operand (not needed for sqrt)",
                    },
                },
                "required": ["operation", "a"],
            },
        },
    }


def tool_function(operation: str, a: float, b: float | None = None) -> str:
    """Execute the calculator tool.

    Args:
        operation: One of add, subtract, multiply, divide, power, sqrt
        a: First operand
        b: Second operand (optional, not needed for sqrt)

    Returns:
        Result as a string, or error message
    """
    try:
        if operation == "add":
            if b is None:
                return "Error: 'b' parameter required for addition"
            return str(a + b)
        elif operation == "subtract":
            if b is None:
                return "Error: 'b' parameter required for subtraction"
            return str(a - b)
        elif operation == "multiply":
            if b is None:
                return "Error: 'b' parameter required for multiplication"
            return str(a * b)
        elif operation == "divide":
            if b is None:
                return "Error: 'b' parameter required for division"
            if b == 0:
                return "Error: Division by zero"
            return str(a / b)
        elif operation == "power":
            if b is None:
                return "Error: 'b' parameter required for power"
            return str(a ** b)
        elif operation == "sqrt":
            if a < 0:
                return "Error: Cannot compute square root of negative number"
            return str(math.sqrt(a))
        else:
            return f"Error: Unknown operation '{operation}'"
    except Exception as e:
        return f"Error: {e}"
