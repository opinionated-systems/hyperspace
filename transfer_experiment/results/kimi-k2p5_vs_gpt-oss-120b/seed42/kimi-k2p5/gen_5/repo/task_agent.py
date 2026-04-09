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
    correct_variations = [r'\bcorrect\b', r'\bright\b', r'\btrue\b', r'\bvalid\b', r'\baccurate\b', r'\bperfect\b']
    incorrect_variations = [r'\bincorrect\b', r'\bwrong\b', r'\bfalse\b', r'\binvalid\b', r'\berror\b', r'\bmistake\b', r'\bflawed\b']
    partial_variations = [r'\bpartial\b', r'\bpartly\b', r'\bpartially\b']
    almost_variations = [r'\balmost\b', r'\bnearly\b', r'\bclose\b']
    
    # Check variations with word boundaries for precision
    for var in correct_variations:
        if re.search(var, pred_lower):
            return "Correct"
    for var in incorrect_variations:
        if re.search(var, pred_lower):
            return "Incorrect"
    for var in partial_variations:
        if re.search(var, pred_lower):
            return "Partial"
    for var in almost_variations:
        if re.search(var, pred_lower):
            return "Almost"
    
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
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict"]:
                    if key in result:
                        val = result[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
                        elif isinstance(val, bool):
                            # Handle boolean values
                            return "Correct" if val else "Incorrect"
    
    # Try to find JSON in markdown code blocks
    markdown_json_pattern = re.search(r'```(?:json)?\s*\n?(\{[^}]+\})\n?```', text, re.DOTALL)
    if markdown_json_pattern:
        try:
            json_obj = json.loads(markdown_json_pattern.group(1))
            if isinstance(json_obj, dict):
                for key in ["response", "classification", "answer", "result", "grade", "evaluation", "verdict"]:
                    if key in json_obj:
                        val = json_obj[key]
                        if isinstance(val, str):
                            normalized = _normalize_prediction(val.strip())
                            if normalized:
                                return normalized
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON-like patterns without tags
    json_pattern = re.search(r'\{\s*"(?:response|classification|answer|result|grade|evaluation|verdict)"\s*:\s*"([^"]+)"\s*\}', text, re.IGNORECASE)
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
    ]
    
    text_lower = text.lower()
    for pattern in grading_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
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

## Key Distinctions:
- **Partial vs Almost**: "Partial" means significant progress but major gaps remain (e.g., missing a key lemma or incomplete case analysis). "Almost" means the solution is essentially complete with only minor issues (e.g., a small arithmetic error or unclear notation that doesn't affect the logic).
- **Incorrect vs Partial**: "Incorrect" means fundamental flaws or wrong approach. "Partial" means good approach but incomplete execution.

## Examples:

**Example 1 - Correct:**
Problem: Prove that the sum of two even numbers is even.
Student: Let a = 2m and b = 2n for integers m, n. Then a + b = 2m + 2n = 2(m+n), which is even.
Classification: Correct (complete proof with proper reasoning)

**Example 2 - Incorrect:**
Problem: Prove that the sum of two even numbers is even.
Student: Even numbers are divisible by 2, so their sum is divisible by 4.
Classification: Incorrect (fundamental error - sum of two evens is not necessarily divisible by 4)

**Example 3 - Partial:**
Problem: Prove that for any prime p > 3, p² ≡ 1 (mod 24).
Student: Any prime p > 3 is odd, so p² ≡ 1 (mod 8). Also, p is not divisible by 3, so p² ≡ 1 (mod 3).
Classification: Partial (correctly identified key facts but didn't combine them to get mod 24)

**Example 4 - Almost:**
Problem: Find all integer solutions to x² + y² = 25.
Student: The solutions are (0,±5), (±5,0), (±3,±4), (±4,±3). There are 12 solutions total.
Classification: Almost (correctly found all solutions but miscounted - there are actually 12 solutions: (0,5), (0,-5), (5,0), (-5,0), (3,4), (3,-4), (-3,4), (-3,-4), (4,3), (4,-3), (-4,3), (-4,-3) - wait, that's actually 12. Let me reconsider: this would be Correct if the count is right)

## Analysis Steps:
1. Compare the student's approach to the official solution
2. Check if key lemmas and theorems are correctly stated and proven
3. Verify if the logic flow is sound and complete
4. Identify any gaps, errors, or missing steps in reasoning
5. Assess whether errors are fundamental (Incorrect), significant gaps (Partial), or minor (Almost)
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
