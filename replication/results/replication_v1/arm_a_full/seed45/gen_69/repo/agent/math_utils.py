"""
Mathematical utilities for comparing expressions and validating answers.

Provides helper functions for IMO grading to determine if student answers
match official solutions, even when expressed in different forms.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in a string."""
    return " ".join(text.split())


def extract_numeric_answer(text: str) -> str | None:
    """Extract a numeric answer from text.
    
    Looks for patterns like:
    - "The answer is 42"
    - "x = 5"
    - "Result: 3/4"
    - "= 7"
    - "Answer: 1 1/2" (mixed numbers)
    - "The value is 3.14e-10" (scientific notation)
    - "Probability: 50%"
    
    Returns the extracted numeric value as a string, or None if not found.
    """
    if not text:
        return None
    
    # First, try to find mixed numbers (e.g., "1 1/2", "2 3/4")
    mixed_pattern = r'(?:the\s+)?(?:final\s+)?(?:answer\s*(?:is|:)\s*)?((?:-?\d+)\s+\d+/\d+)'
    match = re.search(mixed_pattern, text.lower())
    if match:
        return match.group(1).replace(" ", " ")  # Keep space for mixed number parsing
    
    patterns = [
        # "The answer is X" or "Answer: X" - with fraction support
        r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)',
        # "x = 5" or "result = 3.14" - with scientific notation
        r'(?:=|:)\s*([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)',
        # "x=5" without spaces
        r'[xynkm]\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
        # "is 42" at end of sentence
        r'is\s+([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)\s*[.!?]?$',
        # Standalone number at end (with context)
        r'result(?:s)?\s+(?:is|are|:)?\s*([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)',
        # Percentage patterns
        r'(?:value|probability|percentage)\s*(?:is|:)?\s*([+-]?\d+(?:\.\d+)?)\s*%',
        # "equals X" pattern
        r'equals?\s+([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)',
        # "yields X" pattern
        r'yield(?:s|ing)?\s+([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)',
        # "gives X" pattern
        r'gives?\s+([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).replace(" ", "")
    
    # Last resort: find any standalone number that looks like an answer
    # Look for numbers at the end of the text
    last_number_pattern = r'([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)\s*[.!?]?\s*$'
    match = re.search(last_number_pattern, text_lower)
    if match:
        return match.group(1).replace(" ", "")
    
    return None


def parse_fraction(text: str) -> Fraction | None:
    """Parse a fraction from text.
    
    Handles formats like:
    - "3/4"
    - "1 / 2"
    - "5" (treated as 5/1)
    - "0.5" (converted to 1/2)
    
    Returns a Fraction object or None if parsing fails.
    """
    if not text:
        return None
    
    text = text.strip().replace(" ", "")
    
    try:
        # Try direct Fraction parsing
        return Fraction(text)
    except ValueError:
        pass
    
    # Try decimal conversion
    try:
        return Fraction(float(text)).limit_denominator(1000)
    except ValueError:
        pass
    
    return None


def compare_numeric_values(val1: str, val2: str, tolerance: float = 1e-9) -> bool:
    """Compare two numeric values for equality.
    
    Handles:
    - Integers: "42" == "42"
    - Fractions: "3/4" == "0.75"
    - Decimals: "3.14" ≈ "3.14159" (within tolerance)
    - Scientific notation: "1e3" == "1000"
    - Percentages: "50%" == "0.5" == "1/2"
    - Mixed numbers: "1 1/2" == "1.5"
    - Signed numbers: "-5" == "-5.0"
    
    Args:
        val1: First numeric value as string
        val2: Second numeric value as string
        tolerance: Maximum relative difference for float comparison
        
    Returns:
        True if values are equal within tolerance
    """
    if not val1 or not val2:
        return False
    
    # Clean and normalize inputs
    def normalize_value(v: str) -> str:
        v = v.strip().lower()
        # Remove common wrappers
        v = re.sub(r'^[\s=:$]+', '', v)
        v = re.sub(r'[\s.!?;:,]+$', '', v)
        # Handle percentages
        if v.endswith('%'):
            try:
                return str(float(v[:-1]) / 100)
            except ValueError:
                pass
        return v
    
    val1 = normalize_value(val1)
    val2 = normalize_value(val2)
    
    # Handle mixed numbers like "1 1/2"
    mixed_pattern = r'^(-?\d+)\s+(\d+)/(\d+)$'
    for val in [val1, val2]:
        match = re.match(mixed_pattern, val)
        if match:
            whole, num, denom = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if denom != 0:
                if val == val1:
                    val1 = str(whole + num / denom) if whole >= 0 else str(whole - num / denom)
                else:
                    val2 = str(whole + num / denom) if whole >= 0 else str(whole - num / denom)
    
    # Try fraction comparison first (exact)
    frac1 = parse_fraction(val1)
    frac2 = parse_fraction(val2)
    
    if frac1 is not None and frac2 is not None:
        return frac1 == frac2
    
    # Try float comparison with tolerance (handles scientific notation)
    try:
        f1 = float(val1)
        f2 = float(val2)
        
        # Handle special cases
        if f1 == f2:
            return True
        
        # Both zero (handles -0.0 vs 0.0)
        if f1 == 0 and f2 == 0:
            return True
        
        # Check for infinity
        if abs(f1) == float('inf') or abs(f2) == float('inf'):
            return f1 == f2
        
        # Relative tolerance for large numbers
        max_val = max(abs(f1), abs(f2), 1.0)
        if abs(f1 - f2) <= tolerance * max_val:
            return True
        
        # Absolute tolerance for small numbers
        if abs(f1 - f2) <= tolerance:
            return True
        
        # For very large numbers, use a more relaxed relative tolerance
        if max_val > 1e10:
            if abs(f1 - f2) <= 1e-6 * max_val:
                return True
            
    except ValueError:
        pass
    
    return False


def extract_final_answer_section(text: str) -> str:
    """Extract the final answer section from a solution.
    
    Looks for common markers like:
    - "Therefore, the answer is..."
    - "Thus, x = ..."
    - "Final answer: ..."
    - "In conclusion,..."
    
    Returns the extracted section or the last sentence if no marker found.
    """
    if not text:
        return ""
    
    # Look for explicit answer markers
    markers = [
        r'(?:therefore|thus|hence|so|consequently),?\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)',
        r'(?:in\s+)?conclusion,?\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)',
        r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)',
        r'we\s+(?:have|get|obtain)\s*:?\s*',
    ]
    
    text_lower = text.lower()
    for marker in markers:
        match = re.search(marker + r'(.*?)(?:\n|$)', text_lower, re.DOTALL)
        if match:
            # Get the original text at this position
            start = match.start()
            end = match.end()
            return text[start:end].strip()
    
    # Fallback: return last sentence
    sentences = re.split(r'[.!?]+', text)
    if sentences:
        return sentences[-1].strip()
    
    return text[-200:] if len(text) > 200 else text


def are_expressions_equivalent(expr1: str, expr2: str) -> bool:
    """Check if two mathematical expressions are equivalent.
    
    Performs normalization and comparison:
    - Case insensitive
    - Whitespace normalized
    - Common equivalent forms recognized
    
    Args:
        expr1: First expression
        expr2: Second expression
        
    Returns:
        True if expressions appear equivalent
    """
    if not expr1 or not expr2:
        return False
    
    # Normalize both expressions
    norm1 = normalize_expression(expr1)
    norm2 = normalize_expression(expr2)
    
    # Direct comparison
    if norm1 == norm2:
        return True
    
    # Try numeric comparison
    num1 = extract_numeric_answer(expr1)
    num2 = extract_numeric_answer(expr2)
    
    if num1 and num2:
        return compare_numeric_values(num1, num2)
    
    return False


def normalize_expression(expr: str) -> str:
    """Normalize a mathematical expression for comparison.
    
    Normalizations applied:
    - Lowercase
    - Whitespace normalized
    - Remove common wrapper text
    - Standardize multiplication symbols
    """
    if not expr:
        return ""
    
    # Lowercase and normalize whitespace
    expr = expr.lower().strip()
    expr = " ".join(expr.split())
    
    # Remove common wrapper phrases
    wrappers = [
        r'^the answer is\s*',
        r'^answer[:=]?\s*',
        r'^result[:=]?\s*',
        r'^x\s*=\s*',
        r'^y\s*=\s*',
        r'^z\s*=\s*',
        r'^n\s*=\s*',
        r'^k\s*=\s*',
    ]
    
    for wrapper in wrappers:
        expr = re.sub(wrapper, '', expr)
    
    # Standardize multiplication
    expr = expr.replace('×', '*').replace('⋅', '*').replace('·', '*')
    
    # Standardize division
    expr = expr.replace('÷', '/')
    
    # Remove trailing punctuation
    expr = expr.rstrip('.!?;:,')
    
    return expr.strip()


def check_answer_equivalence(
    official_answer: str,
    student_answer: str,
    problem_type: str = "math"
) -> dict[str, Any]:
    """Comprehensive check for answer equivalence.
    
    Returns a dict with:
    - equivalent: bool
    - confidence: str ("high", "medium", "low")
    - method: str (how equivalence was determined)
    - details: dict with additional info
    
    Args:
        official_answer: The official/correct answer
        student_answer: The student's submitted answer
        problem_type: Type of problem (math, geometry, etc.)
    """
    result = {
        "equivalent": False,
        "confidence": "low",
        "method": "none",
        "details": {}
    }
    
    if not official_answer or not student_answer:
        return result
    
    # Try exact match after normalization
    norm_official = normalize_expression(official_answer)
    norm_student = normalize_expression(student_answer)
    
    if norm_official == norm_student:
        result["equivalent"] = True
        result["confidence"] = "high"
        result["method"] = "exact_normalized_match"
        return result
    
    # Try numeric comparison
    official_num = extract_numeric_answer(official_answer)
    student_num = extract_numeric_answer(student_answer)
    
    if official_num and student_num:
        if compare_numeric_values(official_num, student_num):
            result["equivalent"] = True
            result["confidence"] = "high"
            result["method"] = "numeric_equivalence"
            result["details"] = {
                "official_value": official_num,
                "student_value": student_num,
            }
            return result
    
    # Try extracting final answer sections
    official_final = extract_final_answer_section(official_answer)
    student_final = extract_final_answer_section(student_answer)
    
    if official_final and student_final:
        norm_official_final = normalize_expression(official_final)
        norm_student_final = normalize_expression(student_final)
        
        if norm_official_final == norm_student_final:
            result["equivalent"] = True
            result["confidence"] = "medium"
            result["method"] = "final_section_match"
            return result
        
        # Try numeric comparison of final sections
        official_final_num = extract_numeric_answer(official_final)
        student_final_num = extract_numeric_answer(student_final)
        
        if official_final_num and student_final_num:
            if compare_numeric_values(official_final_num, student_final_num):
                result["equivalent"] = True
                result["confidence"] = "medium"
                result["method"] = "final_section_numeric"
                result["details"] = {
                    "official_value": official_final_num,
                    "student_value": student_final_num,
                }
                return result
    
    # Check for substring match (lower confidence)
    if norm_official in norm_student or norm_student in norm_official:
        if len(norm_official) > 3:  # Avoid false positives with short strings
            result["equivalent"] = True
            result["confidence"] = "low"
            result["method"] = "substring_match"
            return result
    
    return result
