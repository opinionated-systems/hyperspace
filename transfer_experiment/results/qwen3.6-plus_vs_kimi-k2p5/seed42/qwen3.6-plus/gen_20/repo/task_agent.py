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
    3. Raw JSON objects with "response" or "grade" field
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

    # Strategy 4: Raw JSON objects with "response" or "grade" field
    # Look for JSON-like structures: {"response": "..."} or {"grade": "..."}
    for key in ["response", "grade"]:
        pattern = rf'\{{\s*"{key}"\s*:\s*"([^"]+)"\s*\}}'
        matches = re.findall(pattern, text)
        for match in matches:
            results.append({key: match})

    # Strategy 5: Case-insensitive response extraction
    # Look for patterns like "response": "correct" or 'response': 'partial'
    # Only match in the last portion of text to avoid picking up from prompt echoes
    last_portion = text[-3000:] if len(text) > 3000 else text
    pattern = r'["\']?(?:response|grade)["\']?\s*:\s*["\']?(correct|incorrect|partial|almost)["\']?'
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
    json_match = re.search(r'<json>\s*\{\s*"(?:response|grade)"\s*:\s*"([^"]+)"\s*\}\s*</json>', text, re.IGNORECASE)
    if json_match:
        val = json_match.group(1).lower()
        if val in ['correct', 'incorrect', 'partial', 'almost']:
            return val

    # Try code block JSON
    code_match = re.search(r'```(?:json)?\s*\{\s*"(?:response|grade)"\s*:\s*"([^"]+)"\s*\}\s*```', text, re.IGNORECASE | re.DOTALL)
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
        # Check if the line contains a grade keyword as the main content
        for grade in ['correct', 'incorrect', 'partial', 'almost']:
            if line == grade or line.endswith(f': {grade}') or line.endswith(f': "{grade}"'):
                return grade

    return None


class TaskAgent:
    """Task agent that solves a grading task with a single LLM call."""

    def __init__(self, model: str = EVAL_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def forward(self, task: str, msg_history: list | None = None) -> tuple[str, list]:
        """Solve the grading task.

        Args:
            task: the grading task (problem + student solution + guidelines)
            msg_history: previous message history (unused for single-call agent)

        Returns:
            Tuple of (prediction, msg_history)
        """
        instruction = f"""You are an expert mathematics competition grader. Your task is to evaluate a student's solution to a math olympiad-style problem and assign it one of four grades.

PROBLEM AND STUDENT SOLUTION:
{task}

GRADING RUBRIC — You MUST choose exactly ONE of these four grades:

1. "correct" — The solution is COMPLETE and RIGOROUS. All essential steps are present and correct. All cases are handled. No logical gaps. The proof could be published as-is.

2. "almost" — The solution is NEARLY complete. The student has the correct main idea and most of the proof is developed correctly. Only ONE small issue remains: a minor case not handled, a small computational slip, or one minor detail not fully justified. The main proof structure is COMPLETE — the student just needs to fill in one small gap.
   → Key test: "Could the student finish this in 5 more minutes?" If YES → "almost".
   → When in doubt between "correct" and "almost", choose "almost".

3. "partial" — The solution shows MEANINGFUL progress but is FAR from complete. The student has proven some useful lemmas or made correct non-trivial observations, BUT the main proof is still largely missing or has SIGNIFICANT gaps.
   → Key test: "Could the student finish this in 5 more minutes?" If NO → "partial".
   → The student has done SOME correct, non-trivial mathematical work that advances toward the solution.

4. "incorrect" — The solution is FUNDAMENTALLY flawed. Contains critical logical errors, shows NO meaningful progress toward the solution, or the approach is wrong from the start. The student has not demonstrated understanding of the key ideas.

DECISION FLOWCHART (follow in order):
Step 1: Is the solution fundamentally wrong or showing no meaningful progress? → "incorrect"
Step 2: Is the main proof structure COMPLETE with only one small gap? → "almost"
Step 3: Is the solution COMPLETE with no gaps at all? → "correct"
Step 4: Does the solution have meaningful progress but the main proof is NOT done? → "partial"

IMPORTANT RULES:
- Be STRICT. Do not give credit for steps that are not actually present in the student's work.
- Do not assume the student knows something they haven't written down.
- "almost" is NOT a synonym for "partial". "almost" means the proof is essentially done. "partial" means significant work remains.
- If the solution has the right idea but is missing a key case or has a small error, it is "almost", NOT "partial".
- If the solution has correct lemmas but hasn't connected them to complete the proof, it is "partial", NOT "almost".

GRADING PROCESS:
1. First, identify what a COMPLETE solution would require (key steps, cases, lemmas).
2. Compare the student's work against this ideal solution.
3. List what the student got RIGHT and what is MISSING or WRONG.
4. Determine if the main proof structure is complete or not.
5. Use the decision flowchart above to assign the grade.

Respond in JSON format:
<json>
{{
    "response": "correct" | "almost" | "partial" | "incorrect"
}}
</json>

Provide your step-by-step reasoning first, then the JSON response at the very end."""

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

        # Try multiple extraction strategies in order of reliability
        try:
            # Strategy 1: Flexible JSON extraction
            extracted = _extract_json_flexible(response_text)
            if extracted:
                # Prefer the LAST valid JSON block (most likely the final answer)
                for item in reversed(extracted):
                    if isinstance(item, dict):
                        # Check both "response" and "grade" keys
                        for key in ["response", "grade"]:
                            if key in item:
                                val = item[key]
                                if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val.lower()
                                    self.log_fn(f"Extracted prediction via JSON (key={key}): {prediction}")
                                    break
                        if prediction != "None":
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
                    if isinstance(last, dict):
                        for key in ["response", "grade"]:
                            if key in last:
                                val = last[key]
                                if isinstance(val, str) and val.lower() in ["correct", "incorrect", "partial", "almost"]:
                                    prediction = val.lower()
                                    self.log_fn(f"Extracted prediction via legacy JSON (key={key}): {prediction}")
                                    break
            except Exception as e:
                self.log_fn(f"Error in legacy extraction: {e}")

        # Validate the prediction
        if prediction not in ["correct", "incorrect", "partial", "almost"]:
            self.log_fn(f"Invalid or missing prediction: '{prediction}', defaulting to incorrect")
            prediction = "incorrect"

        return str(prediction), msg_history
