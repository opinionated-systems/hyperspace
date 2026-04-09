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
    
    # Priority 2: Look for explicit labels in JSON-like format
    json_patterns = [
        ('"almost"', "almost"),
        ("'almost'", "almost"),
        ('"partial"', "partial"),
        ("'partial'", "partial"),
        ('"incorrect"', "incorrect"),
        ("'incorrect'", "incorrect"),
        ('"correct"', "correct"),
        ("'correct'", "correct"),
    ]
    for pattern, label in json_patterns:
        if pattern in text_lower:
            return label
    
    # Priority 3: Look for labels in <json> blocks or markdown code blocks
    # Find content in <json>...</json> blocks
    json_block_pattern = r'<json>\s*(.*?)\s*</json>'
    json_blocks = re.findall(json_block_pattern, text_lower, re.DOTALL)
    for block in json_blocks:
        for label in ["almost", "partial", "incorrect", "correct"]:
            if f'"response": "{label}"' in block or f'"{label}"' in block:
                return label
    
    # Find content in ```json...``` blocks
    code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    code_blocks = re.findall(code_block_pattern, text_lower, re.DOTALL)
    for block in code_blocks:
        for label in ["almost", "partial", "incorrect", "correct"]:
            if f'"response": "{label}"' in block or f'"{label}"' in block:
                return label
    
    # Priority 4: Look for labels after common indicators
    indicators = ["label:", "grade:", "rating:", "assessment:", "verdict:", "final answer:", "conclusion:", 
                  "decision:", "evaluation:", "result:", "classification:", "category:", "label is",
                  "grade is", "the grade", "my assessment", "i grade this as", "this is", "final label:",
                  "i assign", "the answer is", "this should be"]
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
    
    # Priority 5: Look for labels in the last sentence/line (often where conclusion is)
    lines = text_lower.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) < 150:  # Only consider reasonably short lines (likely labels)
            # Check for labels at end of line
            for label in ["almost", "partial", "incorrect", "correct"]:
                if line.endswith(label) or line.endswith(label + ".") or line.endswith(label + "!"):
                    return label
            # Check for labels at start of line
            for label in ["almost", "partial", "incorrect", "correct"]:
                if line.startswith(label) or line.startswith(label + ":"):
                    return label
            # Check for "is <label>" pattern
            for label in ["almost", "partial", "incorrect", "correct"]:
                if f"is {label}" in line or f"is '{label}'" in line or f'is "{label}"' in line:
                    return label
    
    # Priority 6: Look for standalone labels with word boundaries
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
    
    # Priority 7: Check for keywords in order of specificity as fallback
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
        
        instruction = f"""You are an expert grading agent for {domain}. Your task is to carefully analyze the student's answer and assign exactly one of four labels.

## Problem Statement:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Label Definitions (STRICT INTERPRETATION - BE CONSERVATIVE):

**CORRECT**: The answer is PERFECT with NO issues. Use this ONLY when:
- Final answer matches exactly (or equivalent form)
- ALL steps are valid and logically sound
- NO errors of any kind (not even tiny ones)
- Complete work shown as required
- Student demonstrates FULL mastery

**ALMOST**: The answer has MINOR issues only. Use this when:
- Small arithmetic error (e.g., 15×12=190 instead of 180)
- Minor rounding difference (e.g., 28.26 vs 28.27)
- Forgotten constant of integration (+C) but integral is right
- Trivial notation issue that doesn't affect correctness
- Missing a trivial final step that takes <10 seconds to complete
- The core answer is RIGHT but has cosmetic issues
- Correct final answer with minimal work shown for simple problems

**PARTIAL**: The answer has SIGNIFICANT problems but some valid work. Use this when:
- Missing one of multiple required solutions (e.g., x=2 but not x=-2)
- Incomplete proof with gaps in reasoning
- Major calculation error but some correct method shown
- Correct approach but wrong final answer
- Missing significant work or key explanations
- Student shows SOME understanding but not mastery
- Partial progress on a multi-step problem

**INCORRECT**: The answer is fundamentally wrong. Use this when:
- Wrong method or approach entirely
- No valid mathematical reasoning shown
- Answer is just a guess with no work
- Fundamental misunderstanding of concepts
- Answer doesn't address the problem asked
- No meaningful progress toward solution
- Only trivial or irrelevant work shown

## CRITICAL DISTINCTION RULES:

**Correct vs Almost:**
- If there's ANY error (even tiny), it's NOT correct → use ALMOST
- Correct means PERFECT, not "mostly right"
- When in doubt, choose ALMOST over CORRECT
- For simple problems, correct answer with no work shown → ALMOST (not CORRECT)

**Almost vs Partial:**
- ALMOST = minor issues that don't affect the answer's correctness
- PARTIAL = issues that DO affect correctness significantly
- The "10-second test": Can they fix it in 10 seconds? Yes → ALMOST, No → PARTIAL
- Missing solutions = PARTIAL (not ALMOST)
- Correct final answer but missing substantial work → PARTIAL (not ALMOST)

**Partial vs Incorrect:**
- PARTIAL = some valid work and understanding shown
- INCORRECT = no valid work or fundamental misunderstanding
- If they got ANYTHING right meaningfully → PARTIAL
- If it's all wrong or nonsense → INCORRECT
- When in doubt between PARTIAL and INCORRECT, choose PARTIAL if any valid reasoning exists

## Decision Flowchart:
1. Is the answer PERFECT (no issues at all)? → CORRECT
2. Is it fundamentally wrong with no valid work? → INCORRECT
3. Is there some valid work but major problems? → PARTIAL
4. Is it nearly perfect with only tiny issues? → ALMOST

## KEY PITFALLS TO AVOID:
- Don't overuse CORRECT - most answers have at least minor issues
- Don't underuse ALMOST - many good answers with tiny flaws deserve this
- Don't be too harsh on INCORRECT - if they showed any valid reasoning, use PARTIAL
- Missing work is usually PARTIAL, not ALMOST (unless the problem is trivial)

## Common Edge Cases:
- Equivalent forms: (x-3)(x+3) vs (x+3)(x-3) are both CORRECT
- Rounding errors: 28.26 vs 28.27 is ALMOST (minor calculation difference)
- Missing +C in integration: ALMOST (minor omission, core concept understood)
- Missing one of two solutions: PARTIAL (significant omission, not ALMOST)
- Correct answer with no work: PARTIAL or INCORRECT (depends on problem)
- Partial proof with gaps: PARTIAL (shows some understanding but incomplete)
- One small arithmetic slip in long calculation: ALMOST
- Wrong formula used: INCORRECT (fundamental error)

## Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

## Response Format:
First, provide your analysis of the student's answer. Then, provide your final label in the exact JSON format below.

Your analysis should:
1. State what the problem is asking
2. Summarize the correct solution approach
3. Evaluate what the student got right
4. Identify any errors or omissions (be specific about severity)
5. Apply the "10-second test" if relevant: Can the error be fixed in 10 seconds?
6. Explain why you chose your final label based on the decision rules above

After your analysis, you MUST provide the final label in this exact format:

<json>
{{
    "response": "correct|almost|partial|incorrect"
}}
</json>

## CRITICAL REMINDERS:
- **CORRECT means PERFECT** - any error at all means use ALMOST or lower
- **ALMOST = tiny issues only** - missing solutions, incomplete proofs, or major gaps are PARTIAL
- **Be CONSERVATIVE** - when in doubt, choose the LOWER grade
- **The response field must contain ONLY one of these four exact words**: correct, almost, partial, or incorrect
- **Do NOT use phrases like "partially correct" or "almost correct" in the JSON** - use the exact label only
- **JSON FORMAT IS MANDATORY** - Always wrap your final answer in <json>...</json> tags
- **DOUBLE-CHECK your label** - Make sure it matches your analysis before outputting"""

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
