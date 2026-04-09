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

    Tries multiple patterns:
    1. <json>...</json> tags
    2. ```json...``` code blocks
    3. Raw JSON objects with "response" field
    4. JSON-like patterns with flexible whitespace
    """
    results = []

    # Strategy 1: <json>...</json> tags
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

    # Strategy 2: ```json...``` code blocks
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue

    # Strategy 3: ```...``` code blocks (without json label)
    pattern = r'```\s*(\{.*?\})\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue

    # Strategy 4: Raw JSON objects with "response" field
    # Look for JSON-like structures: {"response": "..."}
    pattern = r'\{\s*"response"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        results.append({"response": match})

    # Strategy 5: Case-insensitive response extraction
    # Look for patterns like "response": "correct" or 'response': 'partial'
    # Only match in the last portion of text to avoid picking up from prompt echoes
    last_portion = text[-3000:] if len(text) > 3000 else text
    pattern = r'["\']?response["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?'
    matches = re.findall(pattern, last_portion, re.IGNORECASE)
    for match in matches:
        results.append({"response": match.lower()})

    return results or None


def _extract_response_direct(text: str) -> str | None:
    """Direct extraction of response value from text.

    Looks for explicit mentions of the grade in the text, focusing on
    the final answer/conclusion rather than quoted text from the prompt.
    """
    text_lower = text.lower()

    # First, try to find the JSON block and extract from there
    # This is the most reliable source
    json_match = re.search(r'<json>\s*\{\s*"response"\s*:\s*"([^"]+)"\s*\}\s*</json>', text, re.IGNORECASE)
    if json_match:
        val = json_match.group(1).lower()
        if val in ['correct', 'incorrect', 'partial', 'almost']:
            return val

    # Try code block JSON
    code_match = re.search(r'```(?:json)?\s*\{\s*"response"\s*:\s*"([^"]+)"\s*\}\s*```', text, re.IGNORECASE | re.DOTALL)
    if code_match:
        val = code_match.group(1).lower()
        if val in ['correct', 'incorrect', 'partial', 'almost']:
            return val

    # Look for the grade in the LAST part of the response (after reasoning)
    # Split by common conclusion markers
    conclusion_markers = ['therefore', 'thus', 'hence', 'in conclusion', 'final answer', 'final grade', 'my grade', 'my decision', 'verdict', 'grade:', 'decision:']
    
    # Find the last occurrence of any conclusion marker
    last_conclusion_start = 0
    for marker in conclusion_markers:
        idx = text_lower.rfind(marker)
        if idx > last_conclusion_start:
            last_conclusion_start = idx
    
    # Search for grade in the conclusion portion (or last 500 chars if no marker found)
    if last_conclusion_start > 0:
        search_text = text_lower[last_conclusion_start:]
    else:
        search_text = text_lower[-500:]
    
    # Look for grade patterns in the conclusion
    for grade in ['correct', 'incorrect', 'partial', 'almost']:
        # Pattern: grade in quotes
        if f'"{grade}"' in search_text or f"'{grade}'" in search_text:
            return grade
        # Pattern: grade: <word> or decision: <word>
        pattern = rf'(?:grade|decision|verdict|answer)\s*:\s*{grade}'
        if re.search(pattern, search_text):
            return grade
        # Pattern: is <grade> or is "<grade>"
        pattern = rf'\bis\s+(?:")?{grade}(?:")?'
        if re.search(pattern, search_text):
            return grade

    # Fallback: look for grade keywords in the last few lines
    lines = text_lower.strip().split('\n')
    for line in reversed(lines[-10:]):  # Check last 10 lines
        line = line.strip()
        if line in ['correct', 'incorrect', 'partial', 'almost']:
            return line
        for grade in ['correct', 'incorrect', 'partial', 'almost']:
            if re.search(rf'\b{grade}\b', line) and any(kw in line for kw in ['grade', 'decision', 'verdict', 'answer', 'final']):
                return grade

    # Additional fallback: look for the grade in the very last sentence
    # This catches cases where the model writes the grade at the very end
    last_sentence = text_lower.strip().split('.')[-1].strip()
    for grade in ['correct', 'incorrect', 'partial', 'almost']:
        if grade in last_sentence:
            return grade

    return None


class TaskAgent:
    """Task agent that solves IMO grading problems."""

    def __init__(self, model: str = EVAL_MODEL, log_file: str = "") -> None:
        self.model = model
        self.log_fn = logger.info
        self.log_file = log_file
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)

    def __repr__(self) -> str:
        return f"TaskAgent(model={self.model!r}, log_file={self.log_file!r})"

    def forward(self, inputs: dict) -> tuple[str, list[dict]]:
        """Run the task agent on a single grading problem.

        Args:
            inputs: dict with keys 'problem', 'solution', 'grading_guidelines',
                    and 'student_answer'.

        Returns:
            A tuple of (prediction, msg_history) where prediction is one of
            'correct', 'almost', 'partial', or 'incorrect'.
        """
        self.log_fn(f"Starting task agent with model: {self.model}")

        # Extract fields from inputs for clearer prompting
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert IMO mathematics grader. Grade the student's answer using the official grading guidelines.

GRADE DEFINITIONS:
- "correct" (7/7): Solution is COMPLETE and rigorous. All key steps present, all cases handled, no logical gaps.
- "almost" (6/7): Nearly complete. Core proof structure is correct with only minor gaps, missing edge cases, or small computational errors.
- "partial" (1-5/7): Some meaningful progress but far from complete. Proves useful lemmas, makes correct non-trivial observations, or sets up a correct framework.
- "incorrect" (0/7): Fundamentally wrong, critical logical errors, or NO meaningful progress.

═══════════════════════════════════════════════════════════
Problem:
{problem}

Official Solution:
{solution}

Grading Guidelines (YOUR PRIMARY REFERENCE):
{grading_guidelines}

Student's Answer:
{student_answer}
═══════════════════════════════════════════════════════════

GRADING PROCEDURE — Follow these steps carefully:

STEP 1 — Check Partial Criteria:
Read the "(Partial)" section of the Grading Guidelines above. For EACH criterion listed there, check if the student's answer satisfies it.
- If the student satisfies AT LEAST ONE Partial criterion → the grade CANNOT be "incorrect". It must be at least "partial".
- If the student satisfies NONE of the Partial criteria → the grade is "incorrect".
- Be generous: any correct relevant mathematical content (correct setup, first steps, key ideas, lemmas, special cases) counts as partial credit.

STEP 2 — Check Almost Criteria:
Read the "(Almost)" section of the Grading Guidelines. Does the student satisfy these criteria?
- If YES → the grade is "almost" (unless it fully satisfies everything, then "correct").
- "almost" means the main proof structure is complete with only minor issues.

STEP 3 — Check for Correctness:
Is the student's solution COMPLETE and rigorous with NO gaps?
- Are ALL cases from the official solution handled?
- Are there any unjustified claims or leaps in logic?
- Are there any logical errors?
- Does the student verify their answer?
→ If YES to all → "correct"
→ If NO to any → NOT "correct" (must be "almost" or "partial")

STEP 4 — Assign Final Grade:
Based on Steps 1-3, assign exactly ONE grade:
- "correct" if fully complete and rigorous with NO gaps
- "almost" if nearly complete with minor gaps (main idea is there)
- "partial" if some progress but significant work remains (at least one Partial criterion met)
- "incorrect" if NO Partial criteria are met (no meaningful progress)

Respond in JSON format:
<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

Provide your step-by-step reasoning first, then the JSON response."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from response using multiple strategies
        prediction = "None"
        response_text = ""

        # Get the response text from msg_history or use the direct response
        try:
            if msg_history and len(msg_history) > 0:
                if isinstance(msg_history[-1], dict) and "text" in msg_history[-1]:
                    response_text = msg_history[-1]["text"]
                elif isinstance(msg_history[-1], str):
                    response_text = msg_history[-1]
            else:
                response_text = response if isinstance(response, str) else str(response)
        except Exception as e:
            self.log_fn(f"Error getting response text: {e}")
            response_text = str(response) if response else ""

        self.log_fn(f"Raw response text: {response_text[:500]}...")

        # Try multiple extraction strategies
        try:
            # Strategy 1: Flexible JSON extraction
            extracted = _extract_json_flexible(response_text)
            if extracted:
                # Prefer the LAST valid JSON block (most likely the final answer)
                for item in reversed(extracted):
                    if isinstance(item, dict) and "response" in item:
                        val = item["response"]
                        if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial", "almost"]:
                            prediction = val.lower()
                            self.log_fn(f"Extracted prediction via JSON: {prediction}")
                            break
        except Exception as e:
            self.log_fn(f"Error in flexible JSON extraction: {e}")

        # Strategy 2: Direct response extraction if JSON failed
        if prediction == "None":
            try:
                direct = _extract_response_direct(response_text)
                if direct:
                    prediction = direct
                    self.log_fn(f"Extracted prediction via direct extraction: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in direct extraction: {e}")

        # Strategy 3: Legacy extraction as fallback
        if prediction == "None":
            try:
                extracted = _extract_jsons(response_text)
                if extracted and len(extracted) > 0:
                    last = extracted[-1]
                    if isinstance(last, dict) and "response" in last:
                        val = last["response"]
                        if val in ["correct", "incorrect", "partial", "almost"]:
                            prediction = val
                            self.log_fn(f"Extracted prediction via legacy JSON: {prediction}")
            except Exception as e:
                self.log_fn(f"Error in legacy extraction: {e}")

        # Validate the prediction
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            self.log_fn(f"Invalid or missing prediction: '{prediction}', defaulting to incorrect")
            prediction = "incorrect"

        return str(prediction), msg_history
