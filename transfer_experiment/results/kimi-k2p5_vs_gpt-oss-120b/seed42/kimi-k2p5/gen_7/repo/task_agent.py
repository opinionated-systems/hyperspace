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

from agent.llm_client import get_response_from_llm, EVAL_MODEL

logger = logging.getLogger(__name__)


def _extract_jsons(text: str) -> list[dict] | None:
    """Extract JSON objects from <json>...</json> blocks.

    Uses index/rindex to find outermost tag pairs, avoiding the lazy .*?
    regex bug that truncates content with nested braces.
    """
    results = []
    search_from = 0
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
            continue
    return results or None


def _normalize_prediction(prediction: str) -> str | None:
    """Normalize a prediction string to one of the allowed categories.
    
    Handles common variations and misspellings of category names.
    Uses a priority-based approach to handle overlapping terms correctly.
    """
    if not prediction:
        return None
    
    # Normalize the input
    pred_lower = prediction.lower().strip()
    
    # Check for exact matches first (most reliable)
    allowed_categories = ["correct", "incorrect", "partial", "almost"]
    if pred_lower in allowed_categories:
        return pred_lower.capitalize()
    
    # Check for exact match with word boundaries (e.g., "The answer is Correct")
    for cat in allowed_categories:
        if re.search(rf'\b{cat}\b', pred_lower):
            return cat.capitalize()
    
    # Map common variations to standard categories
    # Be careful with overlapping terms - use word boundaries
    # Order matters: check more specific patterns first
    
    # Check for "almost" first (more specific than "correct")
    almost_variations = [
        r'\balmost\b', r'\balmost correct\b', r'\bnearly correct\b', 
        r'\bclose\b', r'\bminor\b', r'\btrivial\b', r'\bsmall error\b', 
        r'\bminor mistake\b', r'\bslight\b', r'\btiny error\b',
        r'\bessentially correct\b', r'\bmostly correct\b', r'\bnearly right\b'
    ]
    for var in almost_variations:
        if re.search(var, pred_lower):
            return "Almost"
    
    # Check for "partial" (more specific than "incorrect")
    partial_variations = [
        r'\bpartial\b', r'\bpartly\b', r'\bpartially\b', r'\bincomplete\b', 
        r'\bsome progress\b', r'\bhalf\b', r'\bmissing\b', r'\bon the right track\b',
        r'\bgood start\b', r'\bsignificant progress\b', r'\bnot complete\b',
        r'\bunfinished\b', r'\bgood approach\b', r'\bcorrect direction\b'
    ]
    for var in partial_variations:
        if re.search(var, pred_lower):
            return "Partial"
    
    # Check for "incorrect"
    incorrect_variations = [
        r'\bincorrect\b', r'\bwrong\b', r'\bfalse\b', r'\binvalid\b', 
        r'\berror\b', r'\bmistake\b', r'\bflawed\b', r'\bfundamental\b',
        r'\bunsalvageable\b', r'\bbroken\b', r'\bnot correct\b',
        r'\bdoes not work\b', r'\bfails\b', r'\bflawed approach\b'
    ]
    for var in incorrect_variations:
        if re.search(var, pred_lower):
            return "Incorrect"
    
    # Check for "correct" last (most general)
    correct_variations = [
        r'\bcorrect\b', r'\bright\b', r'\btrue\b', r'\bvalid\b', 
        r'\baccurate\b', r'\bperfect\b', r'\bcomplete\b', r'\bsound\b',
        r'\bwell done\b', r'\bexcellent\b', r'\bgood\b', r'\bproper\b'
    ]
    for var in correct_variations:
        if re.search(var, pred_lower):
            return "Correct"
    
    return None


def _extract_response_flexible(text: str) -> str | None:
    """Extract response using multiple fallback strategies.
    
    Tries multiple patterns to find the classification:
    1. JSON format with "response" field
    2. JSON format with "classification" field
    3. Direct mention of categories in text
    4. Look for explicit grading statements in the analysis
    5. Markdown code block JSON extraction
    """
    # Try JSON extraction first (from <json> tags)
    json_results = _extract_jsons(text)
    if json_results:
        for result in json_results:
            if isinstance(result, dict):
                # Check for common field names
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category"]:
                    if key in result:
                        val = result[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
                        elif isinstance(val, bool):
                            # Handle boolean values
                            return "Correct" if val else "Incorrect"
    
    # Try to find JSON in markdown code blocks (handle nested braces better)
    markdown_pattern = r'```(?:json)?\s*\n?(\{[\s\S]*?\})\n?```'
    for match in re.finditer(markdown_pattern, text, re.DOTALL):
        try:
            json_obj = json.loads(match.group(1))
            if isinstance(json_obj, dict):
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict", "category"]:
                    if key in json_obj:
                        val = json_obj[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON-like patterns without tags (single line)
    json_pattern = re.search(r'\{\s*"(?:response|classification|answer|result|grade|evaluation|verdict|category)"\s*:\s*"([^"]+)"\s*\}', text, re.IGNORECASE)
    if json_pattern:
        normalized = _normalize_prediction(json_pattern.group(1).strip())
        if normalized:
            return normalized
    
    # Look for explicit grading statements with more specific patterns
    # These patterns look for the final classification decision in the text
    grading_patterns = [
        # Pattern: "The classification is: X" or "Classification: X"
        r'(?:the\s+)?(?:classification|grade|category|result|evaluation|verdict)\s*(?:is\s*[:=]?\s*|[:=]\s*)(correct|incorrect|partial|almost)',
        # Pattern: "I classify this as: X"
        r'(?:i\s+)?(?:classify|grade|rate|evaluate)\s+(?:this|the\s+(?:answer|solution|response))\s*(?:as\s*[:=]?\s*|[:=]\s*)(correct|incorrect|partial|almost)',
        # Pattern: "This is: X" or "The answer is: X"
        r'(?:this|the\s+(?:answer|solution|response))\s+is\s*(?:therefore\s*)?(?:classified\s+as\s*[:=]?\s*|[:=]\s*)(correct|incorrect|partial|almost)',
        # Pattern: "Final classification: X" or "Final answer: X"
        r'final\s+(?:classification|answer|grade|result|evaluation|verdict)\s*[:=]\s*(correct|incorrect|partial|almost)',
        # Pattern: "Therefore, the answer is X"
        r'therefore[,;]?\s+(?:the\s+)?(?:answer|classification|result|evaluation|verdict)\s+is\s*(?:[:=]?\s*)(correct|incorrect|partial|almost)',
        # Pattern: "In conclusion, X"
        r'in\s+conclusion[,;]?\s+(?:the\s+)?(?:answer|classification|result|evaluation|verdict)\s+is\s*(?:[:=]?\s*)(correct|incorrect|partial|almost)',
        # Pattern: "The student answer is X"
        r'(?:the\s+)?(?:student\s+)?(?:answer|solution|response)\s+is\s*(?:[:=]?\s*)(correct|incorrect|partial|almost)',
        # Pattern: "I would classify this as X"
        r'i\s+would\s+(?:classify|grade|rate|evaluate)\s+(?:this|it)\s+as\s*(?:[:=]?\s*)(correct|incorrect|partial|almost)',
        # Pattern: "This should be classified as X"
        r'this\s+should\s+be\s+(?:classified|graded|rated)\s+as\s*(?:[:=]?\s*)(correct|incorrect|partial|almost)',
        # Pattern: "The appropriate classification is X"
        r'the\s+(?:appropriate|correct|proper)\s+(?:classification|grade|category)\s+is\s*(?:[:=]?\s*)(correct|incorrect|partial|almost)',
        # Pattern: "I rate this as X"
        r'i\s+(?:rate|judge|assess)\s+(?:this|it|the\s+(?:answer|solution|response))\s*(?:as\s*[:=]?\s*|[:=]\s*)(correct|incorrect|partial|almost)',
        # Pattern: "This gets a grade of X"
        r'(?:this|it)\s+(?:gets?|receives?)\s+(?:a\s+)?(?:grade|score|mark|classification)\s*(?:of\s*[:=]?\s*|[:=]\s*)(correct|incorrect|partial|almost)',
    ]
    
    text_lower = text.lower()
    for pattern in grading_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
    
    # Look for standalone category mentions at the end of sentences or lines
    # This helps catch cases where the model just says "Partial." or "Almost"
    standalone_patterns = [
        r'(?:^|\n)\s*(correct|incorrect|partial|almost)\s*[.!]?\s*(?:$|\n)',
        r'(?:classification|grade|result)\s*[:\-]?\s*(correct|incorrect|partial|almost)',
        r'\*\*(correct|incorrect|partial|almost)\*\*',  # Bold markdown
        r'\*\*\s*(correct|incorrect|partial|almost)\s*\*\*',  # Bold markdown with spaces
        r'\b(correct|incorrect|partial|almost)\b\s*\([^)]*\)',  # Category with explanation
        r'["\'](correct|incorrect|partial|almost)["\']',  # Quoted category
        r'\b(correct|incorrect|partial|almost)\b[.!?]?\s*$',  # Category at end of text
    ]
    for pattern in standalone_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).capitalize()
    
    # Look for direct category mentions with word boundaries
    categories = ["Correct", "Incorrect", "Partial", "Almost"]
    text_upper = text.upper()
    
    # Check for explicit category statements with more context
    for category in categories:
        # Look for patterns like "The answer is: Correct" or "Classification: Partial"
        patterns = [
            rf'\b{category.upper()}\b',
            rf'(?:is|are|be)\s*[:\-]?\s*{category.upper()}',
            rf'(?:classification|category|grade|result|answer|evaluation|verdict)\s*[:\-]?\s*{category.upper()}',
        ]
        for pattern in patterns:
            if re.search(pattern, text_upper):
                return category
    
    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single problem.

        Args:
            inputs: dict with domain, problem, solution, grading_guidelines, student_answer

        Returns:
            (prediction, msg_history)
        """
        # Extract fields from inputs
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")
        
        instruction = f"""You are an expert mathematics grader for International Mathematical Olympiad (IMO) problems. Your task is to evaluate a student's answer against the official solution and grading guidelines.

## Problem Statement:
```
{problem}
```

## Official Solution:
```
{solution}
```

## Grading Guidelines:
```
{grading_guidelines}
```

## Student Answer:
```
{student_answer}
```

## Classification Categories:

1. **Correct**: The student answer is complete and correct, matching the official solution. All key steps are present and logically sound. The proof is rigorous and complete.

2. **Incorrect**: The student answer is wrong or contains fundamental errors that invalidate the solution. The approach is flawed, the conclusion is wrong, or there are critical logical gaps that cannot be fixed.

3. **Partial**: The student made significant progress but the solution is incomplete or has major gaps. Some key ideas are present but critical steps are missing. The student has demonstrated understanding of the core concepts but hasn't completed the proof.

4. **Almost**: The student answer is nearly correct with only minor mistakes (e.g., small calculation errors, missing edge cases, or minor notation issues). The overall approach and reasoning are sound, and the solution would be correct with trivial fixes.

## Key Distinctions (CRITICAL):

**Partial vs Almost** (Most Important Distinction):
- **Partial**: Significant progress but MAJOR gaps remain. The solution is missing critical components that would require substantial additional work to complete.
  - Examples: Missing a key lemma, incomplete case analysis, only proving one direction of an iff statement, having the right idea but not executing the proof
  
- **Almost**: Solution is ESSENTIALLY COMPLETE with only MINOR issues. The core logic is sound and complete; only trivial fixes needed.
  - Examples: Small arithmetic error (2+2=5), missing one edge case out of many, minor notation issue, small calculation mistake that doesn't affect the overall proof structure

**Incorrect vs Partial**:
- **Incorrect**: Fundamental flaws, wrong approach, or conclusion is wrong. The reasoning is unsalvageable.
- **Partial**: Good approach, correct direction, but incomplete execution. The reasoning shows understanding but needs more work.

## Detailed Examples:

**Example 1 - Correct:**
Problem: Prove that the sum of two even numbers is even.
Student: Let a = 2m and b = 2n for integers m, n. Then a + b = 2m + 2n = 2(m+n), which is even.
Classification: Correct (complete proof with proper reasoning)

**Example 2 - Incorrect (Fundamental error):**
Problem: Prove that the sum of two even numbers is even.
Student: Even numbers are divisible by 2, so their sum is divisible by 4.
Classification: Incorrect (fundamental error - sum of two evens is not necessarily divisible by 4)

**Example 3 - Incorrect (Wrong approach):**
Problem: Prove that √2 is irrational.
Student: √2 ≈ 1.414, which is not an integer, so it's irrational.
Classification: Incorrect (completely wrong approach - approximation doesn't prove irrationality)

**Example 4 - Partial (Major gaps - missing synthesis):**
Problem: Prove that for any prime p > 3, p² ≡ 1 (mod 24).
Student: Any prime p > 3 is odd, so p² ≡ 1 (mod 8). Also, p is not divisible by 3, so p² ≡ 1 (mod 3).
Classification: Partial (correctly identified key facts but didn't combine them using CRT to get mod 24 - missing the final synthesis)

**Example 5 - Partial (Incomplete proof):**
Problem: Prove that every integer n > 1 has a prime factor.
Student: If n is prime, we're done. If n is composite, it has factors.
Classification: Partial (started case analysis but didn't complete the proof for composite case - no induction or descent argument)

**Example 6 - Partial (Good direction, incomplete):**
Problem: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.
Student: Base case: 1 = 1(2)/2 = 1 ✓. For the inductive step, assume true for n=k.
Classification: Partial (correctly set up induction but didn't complete the inductive step - missing the algebra to show it holds for k+1)

**Example 7 - Partial (Missing key lemma):**
Problem: Prove that the angle bisectors of a triangle meet at a single point.
Student: Let the bisectors of angles A and B meet at point I. Then I is equidistant from all three sides.
Classification: Partial (correctly identified the incenter but didn't prove that the third bisector also passes through I - missing the key lemma that I lies on the third bisector)

**Example 8 - Almost (Minor arithmetic error):**
Problem: Find the sum of integers from 1 to 100.
Student: Using the formula n(n+1)/2 with n=100: 100×101/2 = 5051.
Classification: Almost (correct approach and formula, but arithmetic error: 100×101/2 = 5050, not 5051)

**Example 9 - Almost (Missing edge case):**
Problem: Find all positive integer solutions to x² - y² = 1.
Student: (x-y)(x+y) = 1, so x-y = 1 and x+y = 1, giving x=1, y=0. But y must be positive, so no solutions.
Classification: Almost (correctly analyzed the factorization but missed that x-y and x+y could both be -1, giving x=-1, y=0 - though still no positive solutions, the reasoning was incomplete)

**Example 10 - Almost (Minor notation issue):**
Problem: Prove the Pythagorean theorem.
Student: [Correct proof using similar triangles, but uses a and b for both legs and hypotenuses in different triangles without clear distinction]
Classification: Almost (mathematically correct but notation could be clearer - doesn't affect validity)

**Example 11 - Almost (Small calculation error in proof):**
Problem: Prove that the area of a triangle with sides 3, 4, 5 is 6.
Student: Using Heron's formula: s = (3+4+5)/2 = 5. Area = √(5(5-3)(5-4)(5-5)) = √0 = 0.
Classification: Almost (correct formula but calculation error: s = 6, not 5. Area should be √(6×3×2×1) = 6)

## Analysis Steps:
1. Compare the student's approach to the official solution
2. Check if key lemmas and theorems are correctly stated and proven
3. Verify if the logic flow is sound and complete
4. Identify any gaps, errors, or missing steps in reasoning
5. **CRITICAL**: Determine if the solution is essentially complete (Almost) or has major gaps (Partial)
   - Ask: "Would this solution be correct with just a trivial fix?" → If yes, choose Almost
   - Ask: "Does this show good understanding but need substantial work?" → If yes, choose Partial
   - Ask: "Is the approach fundamentally wrong?" → If yes, choose Incorrect
6. Match against the grading guidelines to determine the appropriate category

## Response Format:
You MUST respond with a JSON object in the following format (wrapped in <json> tags):

<json>
{{
    "response": "Correct" | "Incorrect" | "Partial" | "Almost"
}}
</json>

Important: Use exactly one of the four category names (Correct, Incorrect, Partial, Almost) as the value for the "response" field. Be precise in your classification based on the definitions above."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction using flexible extraction
        prediction = "None"
        try:
            response_text = msg_history[-1]["text"]
            extracted = _extract_response_flexible(response_text)
            if extracted:
                prediction = extracted
                self.log_fn(f"Extracted prediction: {prediction}")
            else:
                # Try one more time with normalization on the raw text
                normalized = _normalize_prediction(response_text)
                if normalized:
                    prediction = normalized
                    self.log_fn(f"Normalized prediction: {prediction}")
                else:
                    self.log_fn(f"Could not extract prediction from response: {response_text[:200]}...")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Validate prediction against allowed categories
        allowed_categories = ["Correct", "Incorrect", "Partial", "Almost"]
        if prediction not in allowed_categories:
            self.log_fn(f"Invalid prediction '{prediction}', defaulting to None")
            prediction = "None"

        return str(prediction), msg_history
