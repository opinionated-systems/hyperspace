"""
Code complexity analysis tool: estimate cyclomatic complexity and other metrics.

Provides complexity metrics to help understand code maintainability and
testing difficulty. Uses simple heuristics based on control flow keywords.
"""

from __future__ import annotations

import os
import re


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "analyze_complexity",
            "description": "Analyze code complexity of a Python file. Returns cyclomatic complexity estimate, cognitive complexity score, and function/method counts. Useful for identifying complex code that may need refactoring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the Python file to analyze",
                    },
                },
                "required": ["path"],
            },
        },
    }


def _count_control_flow_keywords(content: str) -> dict:
    """Count control flow keywords that contribute to complexity."""
    # Keywords that increase cyclomatic complexity
    branch_keywords = ['if', 'elif', 'else', 'for', 'while', 'except', 'finally',
                       'with', 'assert', 'and', 'or']
    
    # Count occurrences
    counts = {}
    for keyword in branch_keywords:
        # Use word boundary regex to match whole words only
        pattern = r'\b' + keyword + r'\b'
        counts[keyword] = len(re.findall(pattern, content))
    
    return counts


def _count_functions_and_classes(content: str) -> dict:
    """Count functions, methods, and classes in Python code."""
    # Match function definitions
    functions = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
    
    # Match class definitions
    classes = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
    
    # Match lambda expressions
    lambdas = len(re.findall(r'\blambda\b', content))
    
    # Match list/dict/set comprehensions (cognitive complexity)
    comprehensions = len(re.findall(r'\[.*for.*in.*\]', content))
    comprehensions += len(re.findall(r'\{.*for.*in.*\}', content))
    
    return {
        'functions': functions,
        'classes': classes,
        'lambdas': lambdas,
        'comprehensions': comprehensions,
    }


def _calculate_cyclomatic_complexity(control_counts: dict) -> int:
    """Estimate cyclomatic complexity from control flow counts.
    
    Base complexity is 1, each branch point adds 1.
    """
    # Start with base complexity of 1
    complexity = 1
    
    # Add counts for branching constructs
    complexity += control_counts.get('if', 0)
    complexity += control_counts.get('elif', 0)
    complexity += control_counts.get('for', 0)
    complexity += control_counts.get('while', 0)
    complexity += control_counts.get('except', 0)
    complexity += control_counts.get('finally', 0)
    
    # Boolean operators add complexity for each additional branch
    complexity += control_counts.get('and', 0)
    complexity += control_counts.get('or', 0)
    
    return complexity


def _calculate_cognitive_complexity(control_counts: dict, structures: dict) -> int:
    """Estimate cognitive complexity (simpler heuristic).
    
    Similar to cyclomatic but with different weighting for nesting.
    """
    score = 0
    
    # Each control structure adds 1
    score += control_counts.get('if', 0)
    score += control_counts.get('elif', 0) * 0.5  # elif is slightly less complex
    score += control_counts.get('for', 0)
    score += control_counts.get('while', 0)
    score += control_counts.get('except', 0)
    score += control_counts.get('with', 0)
    score += control_counts.get('assert', 0)
    
    # Boolean operators add 1 each
    score += control_counts.get('and', 0)
    score += control_counts.get('or', 0)
    
    # Comprehensions add cognitive load
    score += structures.get('comprehensions', 0) * 0.5
    score += structures.get('lambdas', 0) * 0.5
    
    return int(score)


def _get_complexity_rating(complexity: int) -> str:
    """Get a human-readable complexity rating."""
    if complexity <= 10:
        return "Low (simple, easy to test)"
    elif complexity <= 20:
        return "Moderate (manageable complexity)"
    elif complexity <= 30:
        return "High (consider refactoring)"
    else:
        return "Very High (needs refactoring)"


def tool_function(path: str) -> str:
    """Analyze code complexity of a Python file.

    Args:
        path: Absolute path to the Python file

    Returns:
        Formatted string with complexity analysis
    """
    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if not os.path.isfile(path):
        return f"Error: Not a file: {path}"

    # Only analyze Python files
    if not path.endswith('.py'):
        return f"Error: Not a Python file: {path}"

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Get metrics
        control_counts = _count_control_flow_keywords(content)
        structures = _count_functions_and_classes(content)
        
        cyclomatic = _calculate_cyclomatic_complexity(control_counts)
        cognitive = _calculate_cognitive_complexity(control_counts, structures)
        
        # Calculate average complexity per function
        avg_per_function = 0
        if structures['functions'] > 0:
            avg_per_function = cyclomatic / structures['functions']

        # Build result
        lines = [
            f"Complexity Analysis: {path}",
            f"",
            f"Cyclomatic Complexity: {cyclomatic}",
            f"  Rating: {_get_complexity_rating(cyclomatic)}",
            f"",
            f"Cognitive Complexity Score: {cognitive}",
            f"",
            f"Code Structure:",
            f"  Functions: {structures['functions']}",
            f"  Classes: {structures['classes']}",
            f"  Lambdas: {structures['lambdas']}",
            f"  Comprehensions: {structures['comprehensions']}",
        ]
        
        if structures['functions'] > 0:
            lines.append(f"  Avg Complexity per Function: {avg_per_function:.1f}")
        
        lines.extend([
            f"",
            f"Control Flow Breakdown:",
            f"  if/elif statements: {control_counts.get('if', 0) + control_counts.get('elif', 0)}",
            f"  loops (for/while): {control_counts.get('for', 0) + control_counts.get('while', 0)}",
            f"  exception handling: {control_counts.get('except', 0) + control_counts.get('finally', 0)}",
            f"  boolean operators (and/or): {control_counts.get('and', 0) + control_counts.get('or', 0)}",
        ])

        return "\n".join(lines)

    except Exception as e:
        return f"Error analyzing complexity: {e}"
