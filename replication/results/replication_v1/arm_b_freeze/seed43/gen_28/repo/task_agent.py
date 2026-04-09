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


def _extract_any_json(text: str) -> list[dict] | None:
    """Fallback JSON extraction that looks for any JSON objects in text."""
    results = []
    # Try to find JSON objects between curly braces
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    obj = json.loads(text[start_idx:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    return results or None


class TaskAgent:
    """Task agent that solves IMO grading problems with chain-of-thought reasoning."""

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
        # Extract task components for better prompting
        domain = inputs.get("domain", "Unknown")
        problem = inputs.get("problem", "")
        solution = inputs.get("solution", "")
        grading_guidelines = inputs.get("grading_guidelines", "")
        student_answer = inputs.get("student_answer", "")

        instruction = f"""You are an expert mathematical grader for International Mathematical Olympiad (IMO) problems.

Your task is to evaluate a student's answer to a mathematics problem and provide a grade.

DOMAIN: {domain}

PROBLEM:
{problem}

OFFICIAL SOLUTION:
{solution}

GRADING GUIDELINES:
{grading_guidelines}

STUDENT'S ANSWER:
{student_answer}

Think step by step following this structured analysis:

1. PROBLEM ANALYSIS
   - Identify the key mathematical concepts being tested
   - Determine what constitutes a complete and correct solution

2. SOLUTION REVIEW
   - Analyze the official solution's approach and key steps
   - Identify critical proof elements or calculations required

3. STUDENT ANSWER EVALUATION
   - Check if the student understood the problem correctly
   - Verify each step of the student's reasoning
   - Identify any errors, gaps, or incorrect assumptions
   - Note any creative or alternative valid approaches

4. GRADING RUBRIC APPLICATION
   - Full marks: Complete, correct solution with proper justification
   - Partial marks: Significant progress with minor gaps or errors
   - Minimal marks: Some understanding but major flaws
   - No marks: Incorrect approach or no meaningful progress

5. FINAL DETERMINATION
   - Assign grade based on the above analysis
   - Provide clear justification for the grade

Respond in JSON format with the following schema:
<json>
{{
    "reasoning": "Your detailed chain-of-thought analysis covering all 5 steps above",
    "grade": "The final grade (e.g., 'Correct', 'Incorrect', 'Partial', or a numeric score like '7/7', '3/7')",
    "confidence": "High/Medium/Low - your confidence in this grading decision"
}}
</json>"""

        response, msg_history, info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
        )

        # Extract prediction from JSON with enhanced fallback mechanisms
        prediction = "None"
        try:
            last_message = msg_history[-1]["text"]
            
            # Try primary extraction method
            extracted = _extract_jsons(last_message)
            
            # Fallback to generic JSON extraction if primary fails
            if extracted is None:
                extracted = _extract_any_json(last_message)
            
            if extracted:
                # Prefer grade field (new schema), then response field (backward compat)
                last_json = extracted[-1]
                if "grade" in last_json:
                    prediction = last_json["grade"]
                elif "response" in last_json:
                    prediction = last_json["response"]
                elif "answer" in last_json:
                    prediction = last_json["answer"]
                elif "result" in last_json:
                    prediction = last_json["result"]
                elif "evaluation" in last_json:
                    prediction = last_json["evaluation"]
                else:
                    # If no known field, use the first string value found
                    for key, value in last_json.items():
                        if isinstance(value, str) and key != "reasoning" and key != "confidence":
                            prediction = value
                            break
                
                # Validate the prediction is not empty or just whitespace
                if prediction and prediction.strip():
                    prediction = prediction.strip()
                else:
                    prediction = "None"
            else:
                self.log_fn("No JSON found in response, attempting text extraction")
                # Last resort: look for grade-like patterns in text
                import re
                grade_patterns = [
                    r'grade[\s]*[:=][\s]*["\']?([^"\'\n]+)["\']?',
                    r'(?:final\s+)?grade[\s]*[:=][\s]*([\w\s/\d]+)',
                    r'(?:score|mark)[\s]*[:=][\s]*([\w\s/\d]+)',
                ]
                for pattern in grade_patterns:
                    match = re.search(pattern, last_message, re.IGNORECASE)
                    if match:
                        prediction = match.group(1).strip()
                        break
        except Exception as e:
            self.log_fn(f"Error extracting prediction: {e}")

        return str(prediction), msg_history
