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

__version__ = "1.7.0"


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
    # The (?:\s*\w+)?\s* part handles optional language identifier after ```
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
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
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

GRADE DEFINITIONS:

**Correct**: The answer is 100% complete and correct with ZERO flaws.
- All key steps from the official solution are present
- The proof is complete with no gaps
- All reasoning is mathematically valid
- Use ONLY when the solution is perfect

**Incorrect**: The answer is fundamentally wrong with no meaningful progress.
- The approach is completely wrong or misguided
- No key insights from the official solution are present
- The reasoning contains fatal flaws
- Use when there is essentially NO progress toward the solution
- IMPORTANT: If the student's answer has significant gaps in critical steps (e.g., "the proof is technical and omitted", "details are left to the reader"), it should be graded as Incorrect, NOT Partial. Partial requires demonstrating specific key insights from the grading guidelines.

**Partial**: The student made SIGNIFICANT progress but the solution is INCOMPLETE.
- Found key insights, invariants, or lemmas from the official solution
- Made meaningful progress toward the solution
- BUT: Missing critical steps, main proof, or completion
- The solution is unfinished despite good progress
- IMPORTANT: The student must demonstrate the specific partial criteria mentioned in the grading guidelines. If the student's approach is completely different and doesn't match the partial criteria, grade as Incorrect.

**Almost**: The solution is NEARLY COMPLETE with only MINOR issues.
- The core proof structure and main reasoning are correct
- Only trivial errors: small calculation mistakes, typos, minor oversights
- The main argument is essentially valid and complete
- Issues are NEGLIGIBLE and don't affect the core logic

---

DECISION FRAMEWORK - Answer these questions in order:

Q1: Does the student show ANY key insight from the official solution or grading guidelines?
   - If NO → Incorrect (no meaningful progress)
   - If YES → Continue to Q2

Q2: Does the student's answer match the specific partial/almost criteria in the grading guidelines?
   - If the student's approach is completely different from the official solution and doesn't demonstrate the key insights mentioned in the grading guidelines → Incorrect
   - If YES → Continue to Q3

Q3: Is the solution essentially complete with only minor issues?
   - If YES → Almost
   - If NO (significant gaps) → Continue to Q4

Q4: Is the solution fully complete and correct?
   - If YES → Correct
   - If NO (incomplete despite progress) → Partial

---

WORK THROUGH THIS STEP BY STEP:

Step 1: List the KEY ELEMENTS required by the official solution
Step 2: Check which elements are PRESENT in the student's answer
Step 3: Identify any ERRORS or GAPS in the student's reasoning
Step 4: Check if the student's answer matches the specific partial/almost criteria in the grading guidelines
Step 5: Classify errors as "fatal/fundamental" vs "minor/negligible"
Step 6: Apply the Decision Framework above
Step 7: Select the final grade

---

Respond in this exact JSON format:
<json>
{{
    "response": "correct" or "incorrect" or "partial" or "almost"
}}
</json>

You must choose exactly ONE of these four labels. Be rigorous and conservative in your evaluation.
"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple strategies
        prediction = "None"
        try:
            extracted = _extract_any_json(response)
            if extracted:
                # Try to find a response field
                for item in extracted:
                    if isinstance(item, dict):
                        if "response" in item:
                            prediction = item["response"]
                            break
                        # Also check for common alternative field names
                        for key in ["grade", "label", "result", "evaluation", "answer"]:
                            if key in item:
                                prediction = item[key]
                                break
                        if prediction != "None":
                            break

            # If no JSON found, try to extract a simple label from the text
            if prediction == "None":
                # Look for common grade labels in the response
                text_lower = response.lower()
                if "correct" in text_lower and "incorrect" not in text_lower:
                    prediction = "correct"
                elif "incorrect" in text_lower or "wrong" in text_lower:
                    prediction = "incorrect"
                elif "partial" in text_lower:
                    prediction = "partial"
                elif "almost" in text_lower:
                    prediction = "almost"
                else:
                    # Use the first line of the response as prediction
                    first_line = response.strip().split('\n')[0][:100]
                    if first_line:
                        prediction = first_line
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
