"""
Mathematical utilities for comparing expressions and validating answers.

Provides helper functions for IMO grading to determine if student answers
match official solutions, even when expressed in different forms.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

# Try to import sympy for enhanced symbolic comparison
# If not available, fall back to basic comparison methods
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


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
    
    Returns the extracted numeric value as a string, or None if not found.
    """
    if not text:
        return None
    
    patterns = [
        # "The answer is X" or "Answer: X"
        r'(?:the\s+)?answer\s*(?:is|:)\s*([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)',
        # "x = 5" or "result = 3.14"
        r'(?:=|:)\s*([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)',
        # "x=5" without spaces
        r'[xynk]\s*=\s*([+-]?\d+(?:\.\d+)?)',
        # "is 42" at end of sentence
        r'is\s+([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*[.!?]?$',
        # Standalone number at end (with context)
        r'result(?:s)?\s+(?:is|are|:)?\s*([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
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
    
    Args:
        val1: First numeric value as string
        val2: Second numeric value as string
        tolerance: Maximum relative difference for float comparison
        
    Returns:
        True if values are equal within tolerance
    """
    if not val1 or not val2:
        return False
    
    # Try fraction comparison first (exact)
    frac1 = parse_fraction(val1)
    frac2 = parse_fraction(val2)
    
    if frac1 is not None and frac2 is not None:
        return frac1 == frac2
    
    # Try float comparison with tolerance
    try:
        f1 = float(val1)
        f2 = float(val2)
        
        if f1 == f2:
            return True
        
        # Relative tolerance for large numbers
        if abs(f1 - f2) <= tolerance * max(abs(f1), abs(f2), 1.0):
            return True
        
        # Absolute tolerance for small numbers
        if abs(f1 - f2) <= tolerance:
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


def compare_symbolic_expressions(expr1: str, expr2: str) -> dict[str, Any]:
    """Compare two mathematical expressions symbolically using sympy.
    
    This provides more sophisticated comparison than simple numeric evaluation,
    handling algebraic equivalence, trigonometric identities, and more.
    
    Args:
        expr1: First mathematical expression as string
        expr2: Second mathematical expression as string
        
    Returns:
        Dict with 'equivalent' (bool), 'confidence' (str), and 'method' (str)
    """
    result = {
        "equivalent": False,
        "confidence": "low",
        "method": "none"
    }
    
    if not SYMPY_AVAILABLE:
        return result
    
    if not expr1 or not expr2:
        return result
    
    try:
        # Clean up expressions for parsing
        # Remove common wrapper text
        clean1 = normalize_expression(expr1)
        clean2 = normalize_expression(expr2)
        
        # Try to parse as sympy expressions
        try:
            sym1 = sp.sympify(clean1)
            sym2 = sp.sympify(clean2)
        except (sp.SympifyError, TypeError, SyntaxError):
            # If direct parsing fails, try with more aggressive cleaning
            # Remove common mathematical phrases
            phrases_to_remove = [
                r'the answer is',
                r'answer[:=]?',
                r'result[:=]?',
                r'we have',
                r'therefore',
                r'thus',
                r'hence',
            ]
            for phrase in phrases_to_remove:
                clean1 = re.sub(phrase, '', clean1, flags=re.IGNORECASE)
                clean2 = re.sub(phrase, '', clean2, flags=re.IGNORECASE)
            clean1 = clean1.strip()
            clean2 = clean2.strip()
            
            try:
                sym1 = sp.sympify(clean1)
                sym2 = sp.sympify(clean2)
            except (sp.SympifyError, TypeError, SyntaxError):
                return result
        
        # Check for exact symbolic equality
        if sym1 == sym2:
            result["equivalent"] = True
            result["confidence"] = "high"
            result["method"] = "symbolic_exact"
            return result
        
        # Check for algebraic equivalence (simplified forms equal)
        simplified1 = sp.simplify(sym1)
        simplified2 = sp.simplify(sym2)
        
        if simplified1 == simplified2:
            result["equivalent"] = True
            result["confidence"] = "high"
            result["method"] = "symbolic_simplified"
            return result
        
        # Check if difference is zero
        diff = sp.simplify(sym1 - sym2)
        if diff == 0:
            result["equivalent"] = True
            result["confidence"] = "high"
            result["method"] = "symbolic_difference_zero"
            return result
        
        # For expressions with free symbols, check if they're equal for random values
        free_symbols = sym1.free_symbols.union(sym2.free_symbols)
        if free_symbols:
            # Substitute random values and compare
            try:
                subs_dict = {}
                for sym in free_symbols:
                    # Use prime numbers as test values
                    subs_dict[sym] = 2  # Simple test value
                
                val1 = float(sym1.subs(subs_dict))
                val2 = float(sym2.subs(subs_dict))
                
                if abs(val1 - val2) < 1e-10:
                    # Try another test value to increase confidence
                    subs_dict2 = {sym: 3 for sym in free_symbols}
                    val1_2 = float(sym1.subs(subs_dict2))
                    val2_2 = float(sym2.subs(subs_dict2))
                    
                    if abs(val1_2 - val2_2) < 1e-10:
                        result["equivalent"] = True
                        result["confidence"] = "medium"
                        result["method"] = "symbolic_numerical_verification"
                        return result
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        
        # Check for trigonometric equivalence
        try:
            trig_diff = sp.trigsimp(sym1 - sym2)
            if trig_diff == 0:
                result["equivalent"] = True
                result["confidence"] = "high"
                result["method"] = "trigonometric_identity"
                return result
        except Exception:
            pass
        
        # Check for logarithmic/exponential equivalence
        try:
            log_diff = sp.logcombine(sym1 - sym2, force=True)
            if sp.simplify(log_diff) == 0:
                result["equivalent"] = True
                result["confidence"] = "high"
                result["method"] = "logarithmic_identity"
                return result
        except Exception:
            pass
        
    except Exception:
        # Any error during symbolic comparison falls back to other methods
        pass
    
    return result


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
    
    # Try symbolic comparison using sympy (if available)
    if SYMPY_AVAILABLE:
        symbolic_result = compare_symbolic_expressions(official_answer, student_answer)
        if symbolic_result["equivalent"]:
            return {
                "equivalent": True,
                "confidence": symbolic_result["confidence"],
                "method": symbolic_result["method"],
                "details": {}
            }
    
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
        
        # Try symbolic comparison of final sections
        if SYMPY_AVAILABLE:
            symbolic_final = compare_symbolic_expressions(official_final, student_final)
            if symbolic_final["equivalent"]:
                return {
                    "equivalent": True,
                    "confidence": symbolic_final["confidence"],
                    "method": f"final_section_{symbolic_final['method']}",
                    "details": {}
                }
        
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
