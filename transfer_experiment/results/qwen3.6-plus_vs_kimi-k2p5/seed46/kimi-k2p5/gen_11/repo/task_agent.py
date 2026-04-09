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

__version__ = "3.0.0"


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
    # Match ```json ... ``` or just ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return results or None


def _extract_json_with_regex(text: str) -> list[dict] | None:
    """Extract JSON objects using regex patterns for common formats."""
    results = []
    
    # Pattern 1: Look for {"response": "Value"} with various quote styles
    pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    # Pattern 2: Look for {'response': 'Value'} with single quotes
    pattern = r"\{\s*'response'\s*:\s*'([^']+)'\s*\}"
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    # Pattern 3: Look for "response": "Value" or 'response': 'Value' (without braces)
    pattern = r'["\']response["\']\s*:\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})
    
    return results or None


def _extract_any_json(text: str) -> list[dict] | None:
    """Extract JSON objects using multiple strategies."""
    # Try <json> tags first (most reliable)
    results = _extract_jsons(text)
    if results:
        return results

    # Try markdown code blocks
    results = _extract_json_from_markdown(text)
    if results:
        return results

    # Try regex patterns for response field
    results = _extract_json_with_regex(text)
    if results:
        return results
    
    # Fallback: look for any JSON-like object with nested structure support
    # Use a more sophisticated approach for nested braces
    results = []
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(text[start:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    
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
- All key steps from the official solution are present
- The proof is complete with no gaps
- All reasoning is mathematically valid
- NO typos, NO calculation errors, NO missing steps, NO logical gaps
- Use ONLY when the solution is PERFECT - if you can find ANY flaw, it's not Correct
- EXAMPLE: A complete proof with all steps verified and no errors whatsoever

**Almost**: The main proof is CORRECT and COMPLETE, with only truly minor issues.
- The core mathematical argument is sound and complete
- All major insights are present and correctly applied
- Issues are cosmetic: typos, minor calculation errors, notation issues, small gaps that are trivial to fix
- KEY TEST: If you fixed the minor issues in 1-2 lines, would the proof be perfect? If YES → Almost
- KEY TEST: Is the main proof structure correct? If YES and only details are wrong → Almost
- EXAMPLE: Correct proof with a small arithmetic error in the final step
- EXAMPLE: Correct proof with a typo in variable naming
- EXAMPLE: Correct proof missing one trivial verification that can be added in one line

**Partial**: The student has made MEANINGFUL PROGRESS but the main proof is INCOMPLETE or has SIGNIFICANT issues.
- The student has found some key insights or made partial progress
- The main proof is missing critical steps OR has significant logical gaps
- The approach is on the right track but incomplete
- KEY TEST: Did the student find at least one key insight from the official solution? If YES → Partial (not Incorrect)
- KEY TEST: Is the main proof structure missing or significantly flawed? If YES → Partial (not Almost)
- EXAMPLE: Student found the key invariant but didn't complete the proof
- EXAMPLE: Student proved one direction of an iff statement but not the other
- EXAMPLE: Student has the right approach but missing crucial verification steps

**Incorrect**: The answer is fundamentally wrong with NO meaningful progress.
- The approach is completely wrong or misguided
- No key insights from the official solution are present
- The reasoning contains fatal flaws
- Use when there is essentially ZERO progress toward the solution
- KEY TEST: If the student only restates the problem, states obvious facts, or provides a wrong approach → Incorrect
- KEY TEST: If the student makes claims without proof or uses circular reasoning → Incorrect
- EXAMPLE: Student claims "the answer is obvious" or uses a completely wrong method
- EXAMPLE: Student restates the problem statement without adding any new reasoning

---

STEP-BY-STEP GRADING PROCESS:

Step 1: Identify the key insights and main proof structure from the official solution
Step 2: Analyze what the student has actually proven vs. what remains
Step 3: Check if the student found any key insights (invariants, key lemmas, main approach)
Step 4: Evaluate if the main proof structure is present and correct
Step 5: Identify any errors or gaps (categorize as minor/cosmetic vs. major/structural)
Step 6: Apply the KEY TESTS below to determine the appropriate grade
Step 7: Double-check your decision by considering adjacent grades
Step 8: Finalize your grade

---

CRITICAL DECISION RULES:

1. Correct vs. Almost: 
   - If you found ANY flaw (even tiny), it is NOT Correct → use Almost
   - Only use Correct for PERFECT solutions

2. Almost vs. Partial:
   - Almost = main proof is RIGHT and COMPLETE, minor issues only
   - Partial = main proof is MISSING or INCOMPLETE, significant gaps exist
   - Ask: "Would fixing minor issues make this perfect?" If YES → Almost, if NO → Partial

3. Partial vs. Incorrect:
   - Partial requires MEANINGFUL PROGRESS (at least one key insight)
   - Incorrect means ZERO meaningful progress
   - Ask: "Did the student find any key insight from the official solution?" If YES → Partial

4. When in doubt, be CONSERVATIVE:
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
