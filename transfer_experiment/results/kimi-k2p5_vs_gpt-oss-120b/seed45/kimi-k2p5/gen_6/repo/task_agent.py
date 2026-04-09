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


def _extract_json_flexible(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies.
    
    Tries multiple extraction methods in order of preference:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Bare JSON objects (starting with { and ending with })
    """
    # Try <json> tags first
    results = _extract_jsons(text)
    if results:
        return results
    
    # Try ```json code blocks
    results = []
    pattern = r'```json\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    if results:
        return results
    
    # Try bare JSON objects using balanced brace matching
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '{':
            depth = 1
            j = i + 1
            in_string = False
            escape_next = False
            while j < n and depth > 0:
                char = text[j]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                j += 1
            if depth == 0:
                candidate = text[i:j]
                try:
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                i = j
                continue
        i += 1
    
    return results or None


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of the four valid categories.
    
    Args:
        prediction: Raw prediction string from LLM
        
    Returns:
        Normalized prediction: "correct", "almost", "partial", or "incorrect"
    """
    if not isinstance(prediction, str):
        return "incorrect"
    
    prediction_lower = prediction.lower().strip()
    
    # Direct matches (exact match takes highest priority)
    valid_categories = ["correct", "almost", "partial", "incorrect"]
    for cat in valid_categories:
        if prediction_lower == cat:
            return cat
    
    # Check for exact word matches (token-level matching)
    words = re.findall(r'\b\w+\b', prediction_lower)
    for word in words:
        if word in valid_categories:
            return word
    
    # Phrase-level matching with priority (order matters!)
    # Priority: almost > partial > incorrect > correct
    # This order prevents "not correct" from being classified as "correct"
    
    # Check for "almost" indicators (highest priority after exact match)
    # Expanded list to catch more "almost" cases
    almost_indicators = [
        "almost", "nearly", "minor mistake", "minor error", "small error",
        "slight error", "tiny mistake", "nearly correct", "mostly correct",
        "minor issue", "small issue", "trivial mistake", "insignificant error",
        "minor flaw", "small flaw", "tiny error", "slight mistake",
        "essentially correct", "fundamentally correct", "correct approach",
        "correct method", "right approach", "right method", "small slip",
        "minor slip", "tiny slip", "nearly right", "mostly right",
        "minor miscalculation", "small miscalculation", "tiny miscalculation",
        "just a small", "only a minor", "minor typo", "small typo",
        "calculation error", "arithmetic error", "minor arithmetic",
        "slight imprecision", "minor imprecision", "nearly perfect",
        "almost perfect", "very close", "minor omission", "small omission"
    ]
    for indicator in almost_indicators:
        if indicator in prediction_lower:
            return "almost"
    
    # Check for "partial" indicators (before incorrect/correct to catch "partially correct")
    partial_indicators = [
        "partial", "incomplete", "partially", "some progress", "meaningful progress",
        "significant gaps", "major gaps", "incomplete solution", "partial solution",
        "partial credit", "some understanding", "partially correct", "half correct",
        "missing parts", "incomplete answer", "fragmented", "some correct",
        "incomplete proof", "partial proof", "started correctly", "on the right track",
        "good start", "some valid", "partly correct", "halfway", "significant portion",
        "major portion missing", "incomplete reasoning", "missing steps"
    ]
    for indicator in partial_indicators:
        if indicator in prediction_lower:
            return "partial"
    
    # Check for "incorrect" indicators (before "correct" to catch negations)
    incorrect_indicators = [
        "incorrect", "wrong", "error", "not correct", "not right", "false",
        "invalid", "no progress", "meaningless", "fundamentally wrong",
        "completely wrong", "totally wrong", "essentially wrong", "no understanding",
        "fundamental misunderstanding", "empty", "trivial", "nonsense",
        "not valid", "invalid approach", "wrong approach", "incorrect approach",
        "no valid", "no meaningful", "failed to", "does not understand",
        "did not understand", "no attempt", "blank", "no answer"
    ]
    for indicator in incorrect_indicators:
        if indicator in prediction_lower:
            return "incorrect"
    
    # Check for "correct" indicators (lowest priority for substrings)
    correct_indicators = [
        "fully correct", "completely correct", "entirely correct", "totally correct",
        "correct solution", "correct answer", "full marks", "full credit",
        "complete solution", "perfect", "excellent", "full understanding",
        "all correct", "everything correct", "correct throughout", "valid solution",
        "valid proof", "correct proof", "sound reasoning", "correct reasoning"
    ]
    for indicator in correct_indicators:
        if indicator in prediction_lower:
            return "correct"
    
    # Fallback: if just "correct" appears without negation
    if "correct" in prediction_lower and "not " not in prediction_lower:
        return "correct"
    
    # Default fallback
    return "incorrect"


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
        # Build a more structured prompt with clear instructions
        domain = inputs.get('domain', 'Mathematics')
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')
        
        instruction = f"""You are an expert grader for {domain} problems. Your task is to evaluate a student's answer against the correct solution and grading guidelines.

## Problem:
{problem}

## Correct Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric - READ CAREFULLY:
You must classify the student's answer into exactly ONE of these four categories. Pay special attention to the distinctions between categories.

### 1. "correct" (7 points) - FULL MARKS
The answer is complete and fully correct. Requirements:
- All key steps of the solution are present and correct
- Mathematical reasoning is sound throughout
- May have at most negligible notation/presentation issues (typos, minor formatting)
- The student demonstrates complete understanding of the problem

### 2. "almost" (6 points) - NEARLY PERFECT
The answer is nearly complete with strong understanding, but has minor mistakes. Requirements:
- Core approach and main result are correct
- Small calculation errors that don't change the final answer
- Missing a minor case that doesn't affect the main conclusion
- Slight imprecision in reasoning that doesn't invalidate the core argument
- The mistakes are MINOR and the solution is essentially correct
- KEY DISTINCTION: The student clearly understood the problem and had the right approach - only tiny errors prevent it from being "correct"

### 3. "partial" (1 point) - MEANINGFUL PROGRESS
The answer shows meaningful progress and understanding of some key concepts, but has significant gaps. Requirements:
- Student understood PART of the problem (not just random guessing)
- Made progress toward the solution (found a lemma, proved a sub-case, identified key insight)
- Has SIGNIFICANT gaps or major errors that prevent completion
- Incomplete solution with meaningful mathematical content
- NOT just wrong - there must be some valid mathematical progress
- KEY DISTINCTION: The student made real progress but the solution is incomplete or has major issues

### 4. "incorrect" (0 points) - NO CREDIT
The answer is essentially wrong or shows no meaningful progress. Indicators:
- Completely wrong approach or answer
- No valid mathematical progress (just random symbols or nonsense)
- Fundamental misunderstanding of the problem
- Trivial or empty response
- Failed to demonstrate any understanding of key concepts

## Critical Distinctions - READ THIS CAREFULLY:
- "almost" vs "partial": This is the MOST IMPORTANT distinction!
  - "almost": The solution is 90-99% complete. Only minor errors (calculation mistakes, typos, missing one small case) prevent it from being perfect.
  - "partial": The solution is 10-50% complete. Significant work is missing or wrong, but there's still some valid mathematical content.
  - ASK YOURSELF: "If I fixed the errors, would this be a complete correct solution?" If YES → "almost". If NO → "partial".

- "partial" vs "incorrect": 
  - "partial": Student showed SOME valid understanding (proved a lemma, identified key insight, made partial progress)
  - "incorrect": No valid mathematical content at all

## Decision Process:
1. First, check if the answer is fully correct → "correct"
2. If not fully correct, ask: "Is this nearly complete with only minor issues?" If the core approach is right and only small fixes are needed → "almost"
3. If not "almost", check if there's meaningful progress despite gaps → "partial"
4. If no meaningful progress → "incorrect"

## Examples:

Example 1 - "correct":
Problem: Find the sum of 1+2+3+...+10.
Student: "Using the formula n(n+1)/2 with n=10: 10*11/2 = 55."
→ Response: "correct" (complete, correct solution)

Example 2 - "almost":
Problem: Find the sum of 1+2+3+...+10.
Student: "Using the formula n(n+1)/2 with n=10: 10*11/2 = 50." (calculation error, should be 55)
→ Response: "almost" (correct method, minor calculation error - if fixed, would be perfect)

Example 3 - "almost":
Problem: Prove that the sum of angles in a triangle is 180°.
Student: "Draw a line parallel to base through the opposite vertex. Using alternate angles, angle A = angle X and angle B = angle Y. Therefore angle A + angle B + angle C = angle X + angle Y + angle C = 180°." (small error in notation but correct reasoning)
→ Response: "almost" (correct proof with minor notation issues)

Example 4 - "partial":
Problem: Prove that for any triangle, the sum of angles is 180°.
Student: "I know that triangles have 3 angles. If I draw a line parallel to one side through the opposite vertex, I can use alternate angles. But I'm not sure how to finish the proof."
→ Response: "partial" (shows understanding of key geometric insight but incomplete proof - significant work missing)

Example 5 - "partial":
Problem: Solve x² - 5x + 6 = 0.
Student: "I can factor this as (x-2)(x-3) = 0. So one solution is x = 2." (missed x = 3)
→ Response: "partial" (correct approach but incomplete - missed one solution)

Example 6 - "incorrect":
Problem: Find the sum of 1+2+3+...+10.
Student: "The answer is 100 because multiplication makes numbers bigger."
→ Response: "incorrect" (fundamentally wrong approach, no valid reasoning)

## Your Task:
Before giving your final answer, analyze the student's solution step by step:
1. What is the correct approach to solve this problem?
2. Did the student use the correct approach?
3. If yes, what errors did they make? Are these minor ("almost") or major ("partial")?
4. If no, did they make any meaningful progress?

Then provide your final classification.

## Output Format:
Respond ONLY in the following JSON format. The response field MUST be exactly one of: "correct", "almost", "partial", or "incorrect" (lowercase, no extra words).

<json>
{{
    "response": "correct"
}}</json>

Replace "correct" with your assessment based on the rubric above."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using flexible extraction
        prediction = "incorrect"  # Default to incorrect for safety
        try:
            extracted = _extract_json_flexible(msg_history[-1]["text"])
            if extracted:
                last_json = extracted[-1]
                if isinstance(last_json, dict) and "response" in last_json:
                    raw_prediction = last_json["response"]
                    prediction = _normalize_prediction(raw_prediction)
                    self.log_fn(f"Raw prediction: {raw_prediction} -> Normalized: {prediction}")
                else:
                    self.log_fn(f"JSON missing 'response' key: {last_json}")
            else:
                # Try to extract directly from text if no JSON found
                text = msg_history[-1]["text"].lower()
                prediction = _normalize_prediction(text)
                self.log_fn(f"No JSON found, extracted from text: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")
            prediction = "incorrect"

        return str(prediction), msg_history
