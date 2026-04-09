"""
Grading utilities for IMO problem evaluation.

Provides helper functions for analyzing student answers and
calculating grading metrics.
"""

from __future__ import annotations

import re
from typing import Any


def extract_numerical_answer(text: str) -> str | None:
    """Extract a numerical answer from text.
    
    Looks for patterns like:
    - "The answer is 42"
    - "x = 5"
    - "result: 3.14"
    - Numbers at the end of the answer
    
    Returns the extracted number as a string, or None if not found.
    """
    # Pattern: "answer is X" or "equals X" or "= X"
    patterns = [
        r'(?:answer|result|value)\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?)',
        r'(?:=|equals|equal to)\s*(-?\d+(?:\.\d+)?)',
        r'\\boxed\{(-?\d+(?:\.\d+)?)\}',
        r'\\text\{(-?\d+(?:\.\d+)?)\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: find the last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    
    return None


def normalize_answer(answer: str | None) -> str | None:
    """Normalize an answer for comparison.
    
    - Strips whitespace
    - Converts to lowercase
    - Removes common formatting characters
    """
    if answer is None:
        return None
    
    normalized = str(answer).strip().lower()
    # Remove common formatting
    normalized = normalized.replace('$', '').replace('\\', '')
    normalized = normalized.replace('^', '').replace('_', '')
    
    return normalized


def compare_answers(student_answer: str, correct_answer: str, tolerance: float = 1e-6) -> bool:
    """Compare two answers with numerical tolerance.
    
    For numerical answers, uses relative tolerance comparison.
    For string answers, uses normalized string comparison.
    
    Args:
        student_answer: The student's answer
        correct_answer: The correct answer
        tolerance: Relative tolerance for numerical comparison
    
    Returns:
        True if answers match, False otherwise
    """
    student_norm = normalize_answer(student_answer)
    correct_norm = normalize_answer(correct_answer)
    
    if student_norm is None or correct_norm is None:
        return student_norm == correct_norm
    
    # Try numerical comparison first
    try:
        student_num = float(student_norm)
        correct_num = float(correct_norm)
        
        # Relative tolerance comparison
        if abs(correct_num) < tolerance:
            return abs(student_num - correct_num) < tolerance
        return abs((student_num - correct_num) / correct_num) < tolerance
    except ValueError:
        pass
    
    # String comparison
    return student_norm == correct_norm


def analyze_solution_structure(solution: str) -> dict[str, Any]:
    """Analyze the structure of a mathematical solution.
    
    Returns a dict with:
    - has_equations: bool
    - has_proof_structure: bool
    - step_count: int (estimated)
    - key_elements: list of detected elements
    """
    result = {
        "has_equations": False,
        "has_proof_structure": False,
        "step_count": 0,
        "key_elements": [],
    }
    
    # Check for equations
    if re.search(r'[=+\-*/^]', solution):
        result["has_equations"] = True
        result["key_elements"].append("equations")
    
    # Check for proof structure
    proof_markers = ['proof', 'theorem', 'lemma', 'therefore', 'thus', 'hence', 'qed']
    if any(marker in solution.lower() for marker in proof_markers):
        result["has_proof_structure"] = True
        result["key_elements"].append("proof_structure")
    
    # Estimate step count from numbered lists or logical breaks
    numbered_steps = len(re.findall(r'(?:^|\n)\s*(?:\d+[.):]|\([\da-z]\))', solution))
    logical_breaks = len(re.findall(r'(?:therefore|thus|hence|so|then|next)', solution.lower()))
    result["step_count"] = max(numbered_steps, logical_breaks, 1)
    
    # Detect other key elements
    if '\\frac' in solution or '/' in solution:
        result["key_elements"].append("fractions")
    if '\\sqrt' in solution or 'sqrt' in solution.lower():
        result["key_elements"].append("roots")
    if re.search(r'\\[a-z]+\{', solution):
        result["key_elements"].append("latex")
    
    return result


def calculate_partial_credit(
    student_answer: str,
    correct_answer: str,
    solution_analysis: dict[str, Any],
) -> float:
    """Calculate a partial credit score (0.0 to 1.0).
    
    This is a heuristic for estimating how close the student
    was to the correct answer, even if not fully correct.
    """
    score = 0.0
    
    # Exact match gets full credit
    if compare_answers(student_answer, correct_answer):
        return 1.0
    
    # Check for numerical extraction
    student_num = extract_numerical_answer(student_answer)
    correct_num = extract_numerical_answer(correct_answer)
    
    if student_num is not None and correct_num is not None:
        try:
            s = float(student_num)
            c = float(correct_num)
            
            # Partial credit based on numerical closeness
            if s == c:
                score = 1.0
            elif c != 0:
                error_ratio = abs((s - c) / c)
                if error_ratio < 0.01:
                    score = 0.9
                elif error_ratio < 0.1:
                    score = 0.7
                elif error_ratio < 0.5:
                    score = 0.5
                else:
                    score = 0.1
            else:
                score = 0.1
        except ValueError:
            pass
    
    # Boost for showing work (having equations)
    if solution_analysis.get("has_equations"):
        score = min(1.0, score + 0.1)
    
    # Boost for proof structure
    if solution_analysis.get("has_proof_structure"):
        score = min(1.0, score + 0.1)
    
    return score
