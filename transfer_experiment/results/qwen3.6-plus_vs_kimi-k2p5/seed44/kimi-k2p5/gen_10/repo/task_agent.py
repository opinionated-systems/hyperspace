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

__version__ = "2.1.0"

# Valid grade labels for normalization
_VALID_GRADES = {"correct", "almost", "partial", "incorrect"}


def _normalize_prediction(raw: str) -> str:
    """Normalize a raw prediction string to one of the valid grade labels.
    
    Uses word boundary matching to avoid false substring matches
    (e.g., "incorrect" should NOT match "correct").
    
    Args:
        raw: Raw prediction string
        
    Returns:
        Capitalized grade label or "None" if not recognized
    """
    cleaned = raw.strip().strip('"').strip("'").lower()
    
    # Strip common prefixes like "grade:", "prediction:", etc.
    cleaned = re.sub(r'^(grade|prediction|result|answer|verdict|evaluation|response)\s*[:=]\s*', '', cleaned).strip()
    
    # Use word-boundary regex to avoid false substring matches
    for grade in _VALID_GRADES:
        if re.search(r'\b' + re.escape(grade) + r'\b', cleaned):
            return grade.capitalize()
    
    return "None"


def _extract_grade_from_reasoning(text: str) -> str | None:
    """Extract grade from reasoning text when JSON extraction fails.
    
    Looks for explicit grade mentions in the reasoning/analysis section.
    """
    text_lower = text.lower()
    
    # Look for explicit grade statements in reasoning
    patterns = [
        r'grade[\s]*:[\s]*(correct|almost|partial|incorrect)',
        r'classification[\s]*:[\s]*(correct|almost|partial|incorrect)',
        r'evaluation[\s]*:[\s]*(correct|almost|partial|incorrect)',
        r'verdict[\s]*:[\s]*(correct|almost|partial|incorrect)',
        r'this answer is (correct|almost|partial|incorrect)',
        r'answer is (correct|almost|partial|incorrect)',
        r'should be graded as (correct|almost|partial|incorrect)',
        r'qualifies as (correct|almost|partial|incorrect)',
        r'falls under (correct|almost|partial|incorrect)',
        r'belongs to (correct|almost|partial|incorrect)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            grade = match.group(1).lower()
            if grade in _VALID_GRADES:
                return grade.capitalize()
    
    return None


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
            # Try to clean up common issues
            try:
                cleaned = re.sub(r',\s*}', '}', inner)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = cleaned.replace("'", '"')
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_from_markdown(text: str) -> list[dict] | None:
    """Extract JSON objects from markdown code blocks as a fallback."""
    results = []
    pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            results.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r',\s*}', '}', match.strip())
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = cleaned.replace("'", '"')
                results.append(json.loads(cleaned))
            except json.JSONDecodeError:
                continue
    return results or None


def _extract_json_braces(text: str) -> list[dict] | None:
    """Extract JSON objects by finding matching braces as a last resort."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            count = 1
            j = i + 1
            while j < len(text) and count > 0:
                if text[j] == '{':
                    count += 1
                elif text[j] == '}':
                    count -= 1
                j += 1
            if count == 0:
                try:
                    results.append(json.loads(text[i:j]))
                except json.JSONDecodeError:
                    try:
                        cleaned = re.sub(r',\s*}', '}', text[i:j])
                        cleaned = re.sub(r',\s*]', ']', cleaned)
                        cleaned = cleaned.replace("'", '"')
                        results.append(json.loads(cleaned))
                    except json.JSONDecodeError:
                        pass
            i = j
        else:
            i += 1
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
        # Build a structured prompt with clearly labeled fields
        domain = inputs.get("domain", "")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer and assign exactly one of four grades: "Correct", "Almost", "Partial", or "Incorrect".

## Problem Statement:
{problem}

## Official Solution:
{solution}

## Grading Guidelines:
{grading_guidelines}

## Student's Answer:
{student_answer}

## Grading Rubric - READ CAREFULLY:

1. "Correct" - The answer is fully correct, complete, and matches the official solution. All key steps and reasoning are present and valid. The solution would receive full marks.

2. "Almost" - The answer demonstrates COMPLETE understanding with only MINOR errors:
   - Small arithmetic/calculation errors in an otherwise correct derivation
   - Typos in notation or variable names
   - Missing trivial final simplification steps
   - The student clearly knows how to solve the problem but made careless mistakes
   - Key insight: If you fixed these minor errors, the solution would be "Correct"
   - The core approach and reasoning are sound throughout

3. "Partial" - The answer shows SOME understanding but has SIGNIFICANT issues:
   - Missing critical proof steps or logical gaps
   - Incomplete solution (stopped halfway or missed key cases)
   - Significant errors in reasoning that affect the conclusion
   - Correct approach but major calculation errors that invalidate the result
   - Key insight: The student is on the right track but needs substantial work

4. "Incorrect" - The answer is fundamentally wrong:
   - Wrong approach or method entirely
   - Fundamental misunderstanding of the problem
   - No meaningful progress toward the solution

## Critical Distinctions - PAY CLOSE ATTENTION:

**"Almost" vs "Partial" - THIS IS THE MOST COMMON ERROR:**
- "Almost" = The solution is essentially complete with only tiny, fixable flaws
  * The student demonstrates full mastery of the solution method
  * Errors are superficial (typos, arithmetic slips, minor notation issues)
  * If you fixed the errors, you'd have a "Correct" answer
  * Example: Correct proof with one small calculation error

- "Partial" = Significant portions are missing or wrong
  * The student shows some understanding but has major gaps
  * Missing critical steps, incomplete reasoning, or major errors
  * Would need substantial work to become correct
  * Example: Correct approach but stopped halfway, or major logical flaw

**"Partial" vs "Incorrect":**
- "Partial" = Genuine understanding of some key concepts (on the right track)
- "Incorrect" = Little to no understanding (wrong approach or no progress)

## Decision Process:
1. First, check if the answer matches the official solution perfectly → "Correct"
2. If not perfect, ask: "Is the approach completely right with only minor errors?" → "Almost"
3. If not "Almost", ask: "Does the student show genuine understanding of key concepts?" → "Partial"
4. If none of the above → "Incorrect"

## IMPORTANT REMINDERS:
- "Almost" is for NEARLY CORRECT answers with minor flaws
- "Partial" is for answers with SIGNIFICANT gaps or errors
- When in doubt between "Almost" and "Partial", choose "Partial" if there are missing critical steps

## Output Format - CRITICAL:
You MUST output ONLY a JSON object wrapped in <json> tags. Do not include any other text, explanations, or markdown formatting outside the JSON tags.

Your response MUST follow this EXACT format:
<json>
{{"grade": "Correct"}}
</json>

OR

<json>
{{"grade": "Almost"}}
</json>

OR

<json>
{{"grade": "Partial"}}
</json>

OR

<json>
{{"grade": "Incorrect"}}
</json>

The value for "grade" must be exactly one of: "Correct", "Almost", "Partial", or "Incorrect" (case-sensitive, with first letter capitalized)."""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON using multiple methods
        prediction = "None"
        try:
            if msg_history and len(msg_history) > 0:
                text = msg_history[-1].get("text", "")

                # Try multiple extraction methods in order of reliability
                extracted = _extract_jsons(text)
                if not extracted:
                    extracted = _extract_json_from_markdown(text)
                if not extracted:
                    extracted = _extract_json_braces(text)

                if extracted:
                    last_extract = extracted[-1]

                    # Try multiple field names in order of preference
                    # Prioritize "grade" since that's what the prompt asks for
                    pred_value = None
                    for field in ["grade", "response", "evaluation", "result", "answer", "verdict", "prediction"]:
                        if field in last_extract:
                            pred_value = last_extract[field]
                            break

                    if pred_value is not None:
                        prediction = _normalize_prediction(str(pred_value))
                        self.log_fn(f"Extracted prediction: {prediction}")
                    else:
                        # Try to find any value that looks like a valid grade
                        for key, value in last_extract.items():
                            if isinstance(value, str):
                                normalized = _normalize_prediction(value)
                                if normalized != "None":
                                    prediction = normalized
                                    self.log_fn(f"Found valid grade in field '{key}': {prediction}")
                                    break
                else:
                    # No JSON found, try to extract grade from reasoning text first
                    grade_from_reasoning = _extract_grade_from_reasoning(text)
                    if grade_from_reasoning:
                        prediction = grade_from_reasoning
                        self.log_fn(f"Extracted grade from reasoning: {prediction}")
                    else:
                        # Fall back to normalizing raw text
                        prediction = _normalize_prediction(text)
                        self.log_fn(f"No JSON found, normalized from text: {prediction}")
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
