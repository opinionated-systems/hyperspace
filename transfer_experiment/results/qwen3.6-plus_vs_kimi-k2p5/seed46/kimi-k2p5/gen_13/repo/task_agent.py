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

__version__ = "4.1.0"


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


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks."""
    results = []
    # Match ```json ... ```, ``` JSON ... ```, or plain ``` ... ``` blocks
    pattern = r'```\s*(?:\w+)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies, from most to least specific."""
    # Try <json> tags first (most specific)
    results = _extract_jsons(text)
    if results:
        return results

    # Try markdown code blocks
    results = _extract_json_from_markdown(text)
    if results:
        return results

    # Try to find JSON objects directly using brace-matching (handles nested JSON)
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Find matching closing brace
            depth = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                results.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1

    if results:
        return results

    # Fallback: try simple regex for flat JSON objects
    results = []
    pattern = r'\{[^{}]*"[^"]+"[^{}]*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            results.append(json.loads(match))
        except json.JSONDecodeError:
            continue

    return results or None


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
        problem = inputs.get('problem', '')
        solution = inputs.get('solution', '')
        grading_guidelines = inputs.get('grading_guidelines', '')
        student_answer = inputs.get('student_answer', '')

        instruction = f"""You are an expert mathematical grader evaluating IMO-style competition problems. Your task is to carefully analyze a student's answer and assign exactly one of four grades.

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

---

GRADE DEFINITIONS - READ CAREFULLY:

**Correct**: The answer is 100% complete and correct with ZERO flaws.
- All key steps from the official solution are present (or equivalent alternative approach)
- The proof is complete with no gaps
- All reasoning is mathematically valid
- Use ONLY when the solution is perfect

**Incorrect**: The answer is fundamentally wrong with no meaningful progress.
- The approach is completely wrong or misguided
- No key insights from the official solution are present
- The reasoning contains critical errors
- Use when there is no substantial correct content

**Partial**: The answer shows meaningful progress but is incomplete or has significant gaps.
- Some key insights from the official solution are present
- There is a valid approach or partial proof structure
- Missing critical steps or contains non-trivial gaps
- May have some correct lemmas or observations
- Use when there is real mathematical progress but not close to complete

**Almost**: The answer is nearly complete with only minor issues.
- Most key steps are present and correct
- The main proof structure is sound
- Only minor gaps, typos, or easily fixable errors
- The core mathematical argument is valid
- Use when the solution is very close to correct but has small flaws

---

STEP-BY-STEP GRADING PROCESS:

1. ANALYZE THE OFFICIAL SOLUTION:
   - Identify the key insights and main proof steps
   - Note any critical lemmas or techniques used

2. ANALYZE THE STUDENT'S ANSWER:
   - Check if the student identified the main approach
   - Look for key insights from the official solution
   - Identify what parts are correct vs incorrect
   - Note any gaps or missing steps

3. COMPARE TO GRADING GUIDELINES:
   - The grading guidelines indicate what partial credit criteria exist
   - Look for specific achievements mentioned in the guidelines

4. MAKE YOUR DECISION:
   - Correct: Only if perfect
   - Almost: Nearly complete, minor issues only
   - Partial: Real progress but significant gaps
   - Incorrect: No meaningful progress or fundamentally wrong

---

DECISION FRAMEWORK:

When in doubt, be CONSERVATIVE:
   - Doubt between Incorrect/Partial → choose Incorrect
   - Doubt between Partial/Almost → choose Partial
   - Doubt between Almost/Correct → choose Almost

---

Respond in this exact JSON format:
<json>
{{
    "response": "Correct" or "Incorrect" or "Partial" or "Almost"
}}
</json>

IMPORTANT: Your response must be valid JSON. Do not include any text outside the JSON tags."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = self._extract_prediction(response)
        return str(prediction), msg_history

    def _extract_prediction(self, response: str) -> str:
        """Extract the grade prediction from the LLM response.
        
        Uses multiple strategies to find a valid grade label.
        """
        prediction = "None"
        
        try:
            # Strategy 1: Extract JSON and look for response field
            extracted = _extract_any_json(response)
            if extracted:
                for item in extracted:
                    if isinstance(item, dict):
                        # Check for response field first
                        if "response" in item:
                            val = item["response"]
                            if isinstance(val, str):
                                prediction = val.strip()
                                break
                        # Check alternative field names
                        for key in ["grade", "label", "result", "evaluation", "answer"]:
                            if key in item:
                                val = item[key]
                                if isinstance(val, str):
                                    prediction = val.strip()
                                    break
                        if prediction != "None":
                            break

            # Normalize to lowercase for comparison
            if prediction != "None":
                prediction = str(prediction).strip().lower()

            # Strategy 2: Pattern matching in text if no JSON found
            if prediction == "None":
                prediction = self._extract_from_text_patterns(response)

        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        # Final validation
        return self._validate_prediction(prediction, response)

    def _extract_from_text_patterns(self, text: str) -> str:
        """Extract grade label using text pattern matching."""
        text_lower = text.lower()
        
        # Define patterns to check in order of specificity
        patterns = [
            # Exact JSON-like patterns (most specific)
            ('"correct"', "correct"),
            ('"incorrect"', "incorrect"),
            ('"partial"', "partial"),
            ('"almost"', "almost"),
            # Response field patterns
            ('response": "correct"', "correct"),
            ('response":"correct"', "correct"),
            ('response": "incorrect"', "incorrect"),
            ('response":"incorrect"', "incorrect"),
            ('response": "partial"', "partial"),
            ('response":"partial"', "partial"),
            ('response": "almost"', "almost"),
            ('response":"almost"', "almost"),
            # Grade field patterns
            ('grade: correct', "correct"),
            ('grade: incorrect', "incorrect"),
            ('grade: partial', "partial"),
            ('grade: almost', "almost"),
        ]
        
        for pattern, label in patterns:
            if pattern in text_lower:
                return label
        
        # Check for standalone words (order matters - check most specific first)
        # Use word boundaries to avoid partial matches
        if re.search(r'\bpartial\b', text_lower):
            return "partial"
        if re.search(r'\balmost\b', text_lower):
            return "almost"
        if re.search(r'\bincorrect\b', text_lower):
            return "incorrect"
        if re.search(r'\bcorrect\b', text_lower):
            return "correct"
        
        return "None"

    def _validate_prediction(self, prediction: str, original_response: str) -> str:
        """Validate and normalize the prediction to a valid label."""
        valid_labels = ["correct", "incorrect", "partial", "almost"]
        
        pred_lower = prediction.lower().strip()
        
        # Direct match
        if pred_lower in valid_labels:
            return pred_lower
        
        # Try to find any valid label as substring
        for label in valid_labels:
            if label in pred_lower:
                return label
        
        # Log the issue and default to incorrect
        self.log_fn(f"Could not determine valid label from: '{prediction}'")
        self.log_fn(f"Original response: {original_response[:200]}...")
        return "incorrect"
