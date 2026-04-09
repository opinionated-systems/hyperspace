"""
Task agent: solves a given task with a single LLM call.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface, same JSON output format, same extraction logic.

This is the INITIAL task agent. The meta agent modifies this file
during self-improvement. The evaluation harness loads whatever
task_agent.py exists at the agent's repo path.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 3
# Delay between retries in seconds
RETRY_DELAY = 1.0


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    Also handles markdown code blocks with json.
    """
    results = []
    search_from = 0
    
    # First try to find <json>...</json> blocks
    while True:
        start = text.find("<json>", search_from)
        if start == -1:
            break
        end = text.find("</json>", start)
        if end == -1:
            break
        inner = text[start + 6:end].strip()
        search_from = end + 7
        try:
            results.append(json.loads(inner))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Try extracting just the response field if it's a simple object
                if '"response"' in inner or "'response'" in inner:
                    # Look for the response value
                    for quote in ['"', "'"]:
                        pattern = f'"response"{quote}:\s*{quote}'
                        if pattern.replace('"', quote) in inner:
                            # Extract the value after response:
                            parts = inner.split(f'response{quote}:')
                            if len(parts) > 1:
                                value_part = parts[1].strip()
                                # Extract the value
                                if value_part.startswith('"') or value_part.startswith("'"):
                                    end_quote = value_part[1:].find(value_part[0])
                                    if end_quote != -1:
                                        value = value_part[1:end_quote+1]
                                        results.append({"response": value})
                                        break
            except Exception:
                pass
            continue
    
    # Also try markdown code blocks
    if not results:
        search_from = 0
        while True:
            start = text.find("```json", search_from)
            if start == -1:
                # Try without language specifier
                start = text.find("```", search_from)
                if start == -1:
                    break
                end_marker = "```"
                inner_start = start + 3
            else:
                end_marker = "```"
                inner_start = start + 7
            
            end = text.find(end_marker, inner_start)
            if end == -1:
                break
            inner = text[inner_start:end].strip()
            search_from = end + len(end_marker)
            try:
                results.append(json.loads(inner))
            except json.JSONDecodeError:
                # Try to extract response field from malformed JSON
                try:
                    if '"response"' in inner:
                        match = re.search(r'"response"\s*:\s*"([^"]+)"', inner)
                        if match:
                            results.append({"response": match.group(1)})
                except Exception:
                    pass
                continue
    
    return results or None


def _get_llm_response_with_retry(
    msg: str,
    model: str,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY
) -> tuple[str, list[dict], dict]:
    """Get LLM response with retry logic for improved reliability.
    
    Args:
        msg: The message to send to the LLM
        model: The model to use
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Tuple of (response, msg_history, info)
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            response, msg_history, info = get_response_from_llm(
                msg=msg,
                model=model,
                msg_history=[],
            )
            logger.info(f"LLM call succeeded on attempt {attempt + 1}")
            return response, msg_history, info
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts: {last_error}")


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid labels.
    
    Args:
        prediction: Raw prediction string from LLM
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not prediction:
        return "incorrect"
    
    # Clean up the prediction
    cleaned = prediction.strip().lower()
    
    # Remove any punctuation or extra whitespace
    cleaned = cleaned.strip(".!?,:;\"'").strip()
    
    # Map to valid labels
    valid_labels = ["correct", "almost", "partial", "incorrect"]
    
    # Check for exact match first
    if cleaned in valid_labels:
        return cleaned
    
    # Split by common separators and check each part
    parts = re.split(r'[\s,;|]+', cleaned)
    for part in parts:
        part = part.strip(".!?,:;\"'")
        if part in valid_labels:
            return part
    
    # Check for JSON-like format with quotes - be strict about matching
    json_patterns = [
        ('"correct"', "correct"),
        ("'correct'", "correct"),
        ('"almost"', "almost"),
        ("'almost'", "almost"),
        ('"partial"', "partial"),
        ("'partial'", "partial"),
        ('"incorrect"', "incorrect"),
        ("'incorrect'", "incorrect"),
    ]
    for pattern, label in json_patterns:
        if pattern in cleaned:
            return label
    
    # Check for partial matches with word boundaries - STRICT ORDER
    # Check more specific terms first to avoid misclassification
    # "almost" is most specific, then "partial", then "incorrect", then "correct"
    
    # Check for "almost" - highest priority after exact match
    if re.search(r'\balmost\b', cleaned):
        return "almost"
    
    # Check for "partial" 
    if re.search(r'\bpartial\b', cleaned):
        return "partial"
    
    # Check for "incorrect" or "wrong"
    if re.search(r'\bincorrect\b', cleaned) or re.search(r'\bwrong\b', cleaned):
        return "incorrect"
    
    # For "correct", be VERY careful - it can appear in phrases like "partially correct", "almost correct"
    # Only match if it's a standalone word and NOT modified by other grade terms
    if re.search(r'\bcorrect\b', cleaned):
        # Check if it's modified by "partial", "almost", or "incorrect" (or their variants)
        # Pattern: word ending in partial/almost/incorrect followed by correct
        # Pattern: correct followed by word starting with partial/almost/incorrect
        modified_pattern = re.search(
            r'\b(partial|almost|incorrect|partly|mostly|nearly|mostly|somewhat)\w*\s+\w*\bcorrect\b|\bcorrect\b\s+\w*\s*\b(partial|almost|incorrect|partly|mostly|nearly|somewhat)\w*', 
            cleaned
        )
        if not modified_pattern:
            return "correct"
    
    # Default to incorrect if no match found
    logger.warning(f"Could not normalize prediction '{prediction}', defaulting to 'incorrect'")
    return "incorrect"


def _extract_label_from_text(text: str) -> str:
    """Extract a valid label from raw text when JSON parsing fails.
    
    Uses a priority-based approach to find the most likely label.
    Priority order: almost > partial > incorrect > correct (most specific first)
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        One of: correct, almost, partial, incorrect
    """
    if not text:
        return "incorrect"
    
    text_lower = text.lower()
    
    # Priority 1: Look for explicit JSON response field patterns (most reliable)
    # These patterns look for the exact output format we requested
    json_response_patterns = [
        ('"response": "almost"', "almost"),
        ('"response":"almost"', "almost"),
        ("'response': 'almost'", "almost"),
        ("'response':'almost'", "almost"),
        ('"response": "partial"', "partial"),
        ('"response":"partial"', "partial"),
        ("'response': 'partial'", "partial"),
        ("'response':'partial'", "partial"),
        ('"response": "incorrect"', "incorrect"),
        ('"response":"incorrect"', "incorrect"),
        ("'response': 'incorrect'", "incorrect"),
        ("'response':'incorrect'", "incorrect"),
        ('"response": "correct"', "correct"),
        ('"response":"correct"', "correct"),
        ("'response': 'correct'", "correct"),
        ("'response':'correct'", "correct"),
    ]
    for pattern, label in json_response_patterns:
        if pattern in text_lower:
            return label
    
    # Priority 2: Look for explicit labels in JSON-like format within <json> blocks
    import re
    json_block_pattern = r'<json>\s*(.*?)\s*</json>'
    json_blocks = re.findall(json_block_pattern, text_lower, re.DOTALL)
    for block in json_blocks:
        # Look for response field in the block
        for label in ["almost", "partial", "incorrect", "correct"]:
            if f'"response": "{label}"' in block or f'"response":"{label}"' in block:
                return label
            if f"'response': '{label}'" in block or f"'response':'{label}'" in block:
                return label
            # Also check for just the quoted label
            if f'"{label}"' in block or f"'{label}'" in block:
                return label
    
    # Priority 4: Look for labels in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    code_blocks = re.findall(code_block_pattern, text_lower, re.DOTALL)
    for block in code_blocks:
        for label in ["almost", "partial", "incorrect", "correct"]:
            if f'"response": "{label}"' in block or f'"response":"{label}"' in block:
                return label
            if f'"{label}"' in block or f"'{label}'" in block:
                return label
    
    # Priority 5: Look for labels after common indicators
    indicators = ["label:", "grade:", "rating:", "assessment:", "verdict:", "final answer:", "conclusion:", 
                  "decision:", "evaluation:", "result:", "classification:", "category:", "label is",
                  "grade is", "the grade", "my assessment", "i grade this as", "this is", "final label:",
                  "i assign", "the answer is", "this should be", "i conclude", "my conclusion",
                  "final assessment:", "final verdict:", "final decision:", "final grade:",
                  "therefore, the grade is", "thus, the grade is", "in conclusion"]
    for indicator in indicators:
        idx = text_lower.find(indicator)
        if idx != -1:
            after = text_lower[idx + len(indicator):].strip()
            # Check for exact match at start
            for label in ["almost", "partial", "incorrect", "correct"]:
                if after.startswith(label):
                    return label
            # Check for quoted labels
            for quote in ['"', "'"]:
                for label in ["almost", "partial", "incorrect", "correct"]:
                    if after.startswith(quote + label + quote):
                        return label
    
    # Priority 6: Look for labels in the last sentence/line (often where conclusion is)
    lines = text_lower.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) < 200:  # Only consider reasonably short lines (likely labels)
            # Check for labels at end of line
            for label in ["almost", "partial", "incorrect", "correct"]:
                if line.endswith(label) or line.endswith(label + ".") or line.endswith(label + "!") or line.endswith(label + "'") or line.endswith(label + '"'):
                    return label
            # Check for labels at start of line
            for label in ["almost", "partial", "incorrect", "correct"]:
                if line.startswith(label) or line.startswith(label + ":"):
                    return label
            # Check for "is <label>" pattern
            for label in ["almost", "partial", "incorrect", "correct"]:
                if f"is {label}" in line or f"is '{label}'" in line or f'is "{label}"' in line:
                    return label
            # Check for "grade of <label>" or "grade: <label>"
            for label in ["almost", "partial", "incorrect", "correct"]:
                if f"grade of {label}" in line or f"grade: {label}" in line:
                    return label
    
    # Priority 7: Look for standalone labels with word boundaries
    # Check in order of specificity: almost > partial > incorrect > correct
    if re.search(r'\balmost\b', text_lower):
        return "almost"
    if re.search(r'\bpartial\b', text_lower):
        return "partial"
    if re.search(r'\bincorrect\b', text_lower) or re.search(r'\bwrong\b', text_lower):
        return "incorrect"
    
    # For "correct", be VERY careful about phrases like "partially correct" or "almost correct"
    if re.search(r'\bcorrect\b', text_lower):
        # Check if it's modified by other grade terms
        modified_pattern = re.search(
            r'\b(partial|almost|incorrect|partly|mostly|nearly|somewhat)\w*\s+\w*\bcorrect\b|\bcorrect\b\s+\w*\s*\b(partial|almost|incorrect|partly|mostly|nearly|somewhat)\w*', 
            text_lower
        )
        if not modified_pattern:
            return "correct"
    
    # Priority 8: Check for keywords in order of specificity as fallback
    if "almost" in text_lower:
        return "almost"
    if "partial" in text_lower:
        return "partial"
    if "incorrect" in text_lower or "wrong" in text_lower or "error" in text_lower:
        return "incorrect"
    if "correct" in text_lower:
        return "correct"
    
    # Default to incorrect
    logger.warning(f"Could not extract label from text, defaulting to 'incorrect'")
    return "incorrect"


# Few-shot examples showing exact label format with reasoning
FEW_SHOT_EXAMPLES = """
Example 1 - CORRECT (Perfect answer, no issues):
Problem: Find the sum of 2 + 3.
Solution: The sum of 2 + 3 is 5.
Student Answer: 5
Analysis: The student's answer matches the correct solution exactly with no errors or omissions. This is PERFECT.
<json>
{"response": "correct"}
</json>

Example 2 - ALMOST (Minor arithmetic error, core answer correct):
Problem: Calculate 15 × 12.
Solution: 15 × 12 = 180
Student Answer: 15 × 12 = 190
Analysis: The student made a small arithmetic error (190 instead of 180), but the method was correct. This is a tiny slip in calculation, not a conceptual error. The answer is nearly correct.
<json>
{"response": "almost"}
</json>

Example 3 - ALMOST (Minor rounding difference):
Problem: Calculate the area of a circle with radius 3.
Solution: Area = πr² = π × 9 = 9π ≈ 28.2743...
Student Answer: Area = π × 3² = 9π = 28.27
Analysis: The student correctly applied the formula and got 9π, but rounded slightly differently (28.27 vs 28.2743...). This is a minor calculation difference, not a conceptual error.
<json>
{"response": "almost"}
</json>

Example 4 - PARTIAL (Missing one of multiple solutions):
Problem: Find all integer solutions to x² = 4.
Solution: x = 2 or x = -2
Student Answer: x = 2
Analysis: The student found one correct solution but missed the other (-2). Missing solutions is a significant gap, not a minor error.
<json>
{"response": "partial"}
</json>

Example 5 - PARTIAL ("Almost correct" description):
Problem: Prove that the sum of angles in a triangle is 180°.
Solution: [Complete geometric proof with diagram]
Student Answer: The student drew the diagram correctly and identified that angles sum to 180°, but the proof had a logical gap in the reasoning. The text says "this is almost correct."
Analysis: Despite the text saying "almost correct," the answer has a logical gap in reasoning. This is partial credit - some valid work but significant issues.
<json>
{"response": "partial"}
</json>

Example 4 - ALMOST (Forgotten constant of integration):
Problem: Find the integral of 2x.
Solution: ∫2x dx = x² + C
Student Answer: x²
Analysis: The student correctly integrated 2x to get x² but forgot the constant of integration (+C). This is a minor omission that doesn't affect the core correctness of the integration.
<json>
{"response": "almost"}
</json>

Example 5 - PARTIAL (Missing one of multiple solutions):
Problem: Solve x² = 4 for x.
Solution: x = 2 or x = -2.
Student Answer: x = 2
Analysis: The student found one correct solution (x = 2) but missed the other solution (x = -2). Missing a solution is a SIGNIFICANT omission, not a minor issue. The answer is incomplete.
<json>
{"response": "partial"}
</json>

Example 6 - PARTIAL (Incomplete system solution):
Problem: Solve the system: x + y = 5, x - y = 1
Solution: Adding equations: 2x = 6, so x = 3. Then y = 2.
Student Answer: x = 3
Analysis: The student correctly found x = 3 but did not find y = 2. The solution is incomplete - they only solved half the problem. This is a significant gap.
<json>
{"response": "partial"}
</json>

Example 7 - PARTIAL (Incomplete proof):
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Solution: [Base case n=1, inductive step with full algebra]
Student Answer: For n=1: 1 = 1(2)/2 = 1 ✓. Assume true for n=k, then for n=k+1: 1+2+...+k+(k+1) = k(k+1)/2 + (k+1).
Analysis: The student correctly proved the base case and set up the inductive step, but didn't complete the algebraic simplification to show it equals (k+1)(k+2)/2. The proof has a significant gap.
<json>
{"response": "partial"}
</json>

Example 8 - INCORRECT (No valid reasoning):
Problem: Prove that the sum of angles in a triangle is 180 degrees.
Solution: [Detailed geometric proof]
Student Answer: It's 180 because that's what I learned.
Analysis: The student states the correct fact but provides no proof or reasoning. No valid mathematical work is shown. This does not demonstrate understanding.
<json>
{"response": "incorrect"}
</json>

Example 9 - INCORRECT (Fundamental error):
Problem: Find the derivative of sin(x).
Solution: cos(x)
Student Answer: -cos(x)
Analysis: The student confused the derivative of sin(x) with the derivative of cos(x). This is a fundamental conceptual error - they used the wrong formula entirely.
<json>
{"response": "incorrect"}
</json>

Example 10 - CORRECT (Equivalent form):
Problem: Factor x² - 9.
Solution: (x + 3)(x - 3)
Student Answer: (x - 3)(x + 3)
Analysis: The student correctly factored the expression. The order of factors doesn't matter - both forms are mathematically equivalent and correct.
<json>
{"response": "correct"}
</json>

Example 11 - ALMOST vs CORRECT distinction:
Problem: Solve 2x + 4 = 10.
Solution: 2x = 6, so x = 3.
Student Answer: x = 3 (with no work shown)
Analysis: While the final answer is correct, no work is shown. For a simple equation this might be acceptable, but strictly speaking, we should see the steps. However, since the problem is trivial and the answer is right, this is ALMOST (not CORRECT because work is missing).
<json>
{"response": "almost"}
</json>

Example 12 - PARTIAL vs ALMOST distinction:
Problem: Find all roots of x³ - x = 0.
Solution: x(x² - 1) = 0, so x = 0, 1, -1
Student Answer: x = 0, x = 1
Analysis: The student found two of the three roots. Missing one root (x = -1) is a significant omission - they didn't fully solve the problem. This is PARTIAL, not ALMOST.
<json>
{"response": "partial"}
</json>

Example 13 - ALMOST (Correct answer with minor presentation issue):
Problem: Find the derivative of f(x) = 3x² + 2x.
Solution: f'(x) = 6x + 2
Student Answer: f'(x) = 6x + 2 (used inconsistent notation, wrote df/dx in one place and f'(x) in another)
Analysis: The student's answer is mathematically correct. The notation inconsistency is a minor presentation issue that doesn't affect the mathematical correctness. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 14 - ALMOST (Correct approach, tiny slip):
Problem: Evaluate ∫(2x + 3) dx from 0 to 2.
Solution: [x² + 3x] from 0 to 2 = (4 + 6) - (0) = 10
Student Answer: ∫(2x + 3) dx = x² + 3x. At x=2: 4 + 6 = 10. At x=0: 0. Final answer: 10.
Analysis: The student correctly integrated and evaluated, but wrote "At x=0: 0" instead of showing the full subtraction. The work is essentially complete with a minor presentation gap. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 15 - INCORRECT (No valid mathematical work):
Problem: Solve the system: 2x + y = 7, x - y = 1
Solution: Adding equations: 3x = 8, so x = 8/3, then y = 5/3
Student Answer: x = 2, y = 3 (guessed without showing any work or reasoning)
Analysis: The student provided an answer with no work shown. While the answer happens to satisfy the first equation (2*2 + 3 = 7), it doesn't satisfy the second (2 - 3 ≠ 1). No valid reasoning is demonstrated. This is INCORRECT.
<json>
{"response": "incorrect"}
</json>

Example 16 - PARTIAL (Some valid reasoning, wrong conclusion):
Problem: Prove that the sum of two odd numbers is even.
Solution: Let the numbers be 2k+1 and 2m+1. Sum = 2k+1+2m+1 = 2(k+m+1), which is even.
Student Answer: Let the numbers be 2k+1 and 2m+1. Adding them: 2k+1+2m+1 = 2k+2m+2. This equals 2(k+m+1) which is divisible by 2, so it's even.
Analysis: The student correctly set up the problem and performed the algebra correctly. The reasoning is valid and leads to the correct conclusion. Wait - actually this looks correct. Let me reconsider: The student showed valid work and reached the correct conclusion. This should be CORRECT or ALMOST depending on whether we consider the setup complete. Given the clear reasoning and correct conclusion, this is CORRECT.
<json>
{"response": "correct"}
</json>

Example 17 - PARTIAL (Good start, incomplete finish):
Problem: Find the maximum value of f(x) = -x² + 4x on [0, 4].
Solution: f'(x) = -2x + 4 = 0 → x = 2. f(2) = -4 + 8 = 4. Check endpoints: f(0) = 0, f(4) = 0. Maximum is 4.
Student Answer: f'(x) = -2x + 4. Setting to 0: x = 2.
Analysis: The student correctly found the critical point but didn't evaluate the function at that point or check endpoints. The solution is incomplete - they only did the first step. This is PARTIAL.
<json>
{"response": "partial"}
</json>

Example 18 - ALMOST (Correct with minor error in final step):
Problem: Simplify (x² - 9)/(x - 3) for x ≠ 3.
Solution: (x² - 9)/(x - 3) = (x+3)(x-3)/(x-3) = x + 3
Student Answer: (x² - 9)/(x - 3) = (x+3)(x-3)/(x-3) = x + 3 for x ≠ 3
Analysis: The student correctly factored, canceled, and stated the domain restriction. The answer is correct. This is CORRECT.
<json>
{"response": "correct"}
</json>

Example 19 - ALMOST (Correct answer, minor typo):
Problem: Compute 15².
Solution: 225
Student Answer: 15² = 225 (wrote 15² as "fifteen sqaured" in text but the math is correct)
Analysis: The student made a spelling error ("sqaured" instead of "squared") but the mathematical answer is correct. This is a cosmetic issue only. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 20 - INCORRECT vs PARTIAL distinction:
Problem: Find ∫e^x sin(x) dx.
Solution: Use integration by parts twice and solve.
Student Answer: I think the answer involves e^x and cos(x) somehow.
Analysis: The student shows awareness that the solution involves e^x and trigonometric functions, which is a tiny bit of understanding, but no actual work or valid reasoning is shown. This is on the borderline. Given the complete lack of actual mathematical work, this is INCORRECT.
<json>
{"response": "incorrect"}
</json>

Example 21 - ALMOST (Correct answer with minor sign error in intermediate step):
Problem: Solve x² - 5x + 6 = 0.
Solution: (x-2)(x-3) = 0, so x = 2 or x = 3.
Student Answer: x² - 5x + 6 = 0. Factoring: (x-2)(x-3) = 0. Solutions: x = 2, x = 3. (Note: student wrote -5x as +5x in one intermediate line but corrected it)
Analysis: The student made a tiny sign slip in an intermediate step but immediately corrected it and got the right answer. The final answer is correct and the work shows understanding. This is ALMOST due to the minor slip.
<json>
{"response": "almost"}
</json>

Example 22 - ALMOST (Correct method, tiny calculation error in final answer):
Problem: Find the area of a rectangle with length 8 and width 5.
Solution: Area = 8 × 5 = 40.
Student Answer: Area = 8 × 5 = 40. (Student wrote 8×5=40 but had written 8×5=42 in scratch work, then corrected to 40)
Analysis: The student used the correct method and got the right final answer. The tiny calculation error in scratch work that was immediately corrected is a minor issue. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 23 - ALMOST (Correct answer, minor unit notation issue):
Problem: A car travels 60 miles in 2 hours. What is its speed?
Solution: Speed = distance/time = 60/2 = 30 mph.
Student Answer: Speed = 60/2 = 30 (forgot to write "mph" but the calculation is correct)
Analysis: The student correctly calculated the speed. Forgetting the unit label is a minor presentation issue that doesn't affect the mathematical correctness. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 24 - ALMOST (Correct approach, one small arithmetic slip):
Problem: Calculate 123 + 456.
Solution: 123 + 456 = 579.
Student Answer: 123 + 456 = 578 (off by 1 due to small addition error)
Analysis: The student understood addition perfectly but made a tiny arithmetic slip. The method is completely correct, just a calculation error. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 25 - ALMOST (Correct proof with one trivial typo):
Problem: Prove that the sum of two even numbers is even.
Solution: Let numbers be 2k and 2m. Sum = 2k + 2m = 2(k+m), which is even.
Student Answer: Let the even numbers be 2k and 2m. Their sum is 2k + 2m = 2(k+m). Since k+m is an integer, 2(k+m) is even. (Student wrote "2k" as "2k" consistently but had one instance of "2l" which was clearly a typo for "2k")
Analysis: The proof is complete and correct. One trivial typo that doesn't affect the logic is a minor issue. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 26 - PARTIAL (Good start, incomplete finish - NOT ALMOST):
Problem: Find the equation of the line through (2, 3) with slope 4.
Solution: y - 3 = 4(x - 2), so y = 4x - 5.
Student Answer: Using point-slope form: y - 3 = 4(x - 2).
Analysis: The student correctly identified and applied the point-slope formula but didn't simplify to slope-intercept form. This is incomplete work - they only did half the problem. This is PARTIAL, not ALMOST.
<json>
{"response": "partial"}
</json>

Example 27 - ALMOST (Correct with minor formatting issue):
Problem: Solve 3x = 12.
Solution: x = 4.
Student Answer: 3x = 12, x=4 (no space after comma, but answer is correct)
Analysis: The answer is mathematically correct. The missing space is a trivial formatting issue. This is ALMOST.
<json>
{"response": "almost"}
</json>

Example 28 - ALMOST vs PARTIAL distinction (borderline case):
Problem: Factor x² - 4 completely.
Solution: (x-2)(x+2)
Student Answer: (x-2)(x+2) = x² - 4 (student wrote the factorization correctly but also wrote the expansion, showing they understand the relationship)
Analysis: The student correctly factored the expression. The additional expansion shows understanding, not confusion. The answer is essentially correct with extra (not wrong) information. This is ALMOST (could argue CORRECT, but the extra expansion is slightly unconventional).
<json>
{"response": "almost"}
</json>
"""


class TaskAgent:
    """Task agent that solves IMO grading problems with improved reliability."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.call_count = 0
        self.error_count = 0

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem with retry logic.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        self.call_count += 1
        logger.info(f"TaskAgent forward call #{self.call_count}")
        
        # Extract fields from inputs for better structured prompting
        domain = inputs.get("domain", "Mathematics")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert {domain} grader. Analyze the student's answer and assign ONE label.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student Answer:
{student_answer}

## Labels (choose ONE):
- **correct**: PERFECT. Zero errors. All steps valid. Full mastery demonstrated.
- **almost**: Minor issues only. Small arithmetic slip, tiny rounding diff, forgotten +C, trivial typo. Core answer is RIGHT.
- **partial**: Significant gaps. Missing solutions, incomplete proof, major error but some valid work shown. SOME understanding.
- **incorrect**: Fundamentally wrong. No valid reasoning, wrong method, pure guess. No meaningful progress.

## Classification Decision Tree (follow strictly):
1. Is the answer PERFECT in every way? (zero errors, complete, all steps valid)
   → correct
   
2. Is there a minor flaw that could be fixed in under 30 seconds? (tiny arithmetic slip, small rounding diff, trivial typo, forgotten +C)
   → almost
   
3. Is there significant valid work but also major gaps/errors? (missing solutions, incomplete proof, major error but some understanding shown)
   → partial
   
4. Is the answer fundamentally wrong with no valid reasoning?
   → incorrect

## Critical Rules:
- If ANY non-trivial error exists → NOT correct
- If missing parts of multi-part answer → partial (not almost)
- If correct final answer but no work shown → partial or almost (not correct)
- If wrong method/formula used → incorrect (even if final number matches)
- If started well but didn't finish → partial
- "Almost correct" in text means partial, not almost
- When in doubt between two grades, choose the LOWER one

{FEW_SHOT_EXAMPLES}

## Output Format:
Provide brief analysis, then EXACTLY this JSON:

<json>
{{"response": "correct"}}  or  {{"response": "almost"}}  or  {{"response": "partial"}}  or  {{"response": "incorrect"}}
</json>

Rules: Use ONLY those 4 words. When in doubt, choose LOWER grade."""

        try:
            response, msg_history, info = _get_llm_response_with_retry(
                msg=instruction,
                model=self.model,
            )
        except RuntimeError as e:
            logger.error(f"Failed to get LLM response: {e}")
            self.error_count += 1
            return "Error: Failed to get LLM response", []

        # Extract prediction from JSON with improved error handling
        prediction = "incorrect"  # Default to incorrect if extraction fails
        try:
            if msg_history and len(msg_history) > 0:
                response_text = msg_history[-1].get("text", "")
                
                # Try JSON extraction first
                extracted = _extract_jsons(response_text)
                if extracted and len(extracted) > 0:
                    # Find the last JSON object with a "response" field
                    for json_obj in reversed(extracted):
                        if "response" in json_obj:
                            raw_prediction = json_obj["response"]
                            prediction = _normalize_prediction(raw_prediction)
                            logger.info(f"Successfully extracted prediction from JSON: {prediction}")
                            break
                    else:
                        # No response field found, try text extraction
                        prediction = _extract_label_from_text(response_text)
                        logger.info(f"Extracted label from text (no response field): {prediction}")
                else:
                    # Try to extract label from raw text if JSON parsing fails
                    prediction = _extract_label_from_text(response_text)
                    logger.info(f"Extracted label from text (no JSON): {prediction}")
            else:
                logger.warning("Empty message history from LLM")
        except Exception as e:
            self.error_count += 1
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
    
    def get_stats(self) -> dict:
        """Return agent statistics for monitoring.
        
        Returns:
            Dictionary with call_count and error_count
        """
        return {
            "call_count": self.call_count,
            "error_count": self.error_count,
            "success_rate": (self.call_count - self.error_count) / max(self.call_count, 1)
        }
